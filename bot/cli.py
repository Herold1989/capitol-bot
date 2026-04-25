from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from bot.alerting import send_alert
from bot.capitol_scraper import CapitolTradesScraper
from bot.config import load_config, resolve_config_path
from bot.db import connect as connect_db
from bot.db import ensure_schema
from bot.market_data import MarketDataClient
from bot.paper_trader import PaperTrader
from bot.reporting import (
    load_latest_daily_report,
    load_portfolio_history,
    render_daily_message,
    render_portfolio_chart_svg,
    write_portfolio_chart_png,
)
from bot.routine import build_routine_status, cache_is_fresh, read_cached_frame, write_cached_frame
from bot.strategy import CapitolStrategy, format_metrics


def _build_parser() -> argparse.ArgumentParser:
    # Keep the CLI narrow so the daily VPS command remains easy to audit.
    parser = argparse.ArgumentParser(description="Capitol Trades paper bot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="compute a paper run without writing DB state")
    parser.add_argument("command", choices=["backtest", "paper-run", "daily-report", "health-check", "discord-test"])
    return parser


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _docker_image_id() -> str | None:
    try:
        return Path("/etc/hostname").read_text().strip()
    except Exception:
        return None


def _write_manifest(
    out_dir: Path,
    config_path: str,
    command: str,
    started_at: float,
    routine_status: object,
    disclosures: pd.DataFrame,
    prices: pd.DataFrame,
    status: str,
) -> None:
    resolved_config_path = resolve_config_path(config_path)
    config_bytes = resolved_config_path.read_bytes()
    manifest = {
        "command": command,
        "status": status,
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "config_path": str(resolved_config_path),
        "git_commit": _git_commit_hash(),
        "docker_image_id": _docker_image_id(),
        "disclosure_count": int(len(disclosures)),
        "disclosure_date_range": [
            str(pd.to_datetime(disclosures["published_at"], errors="coerce").min().date()) if not disclosures.empty and "published_at" in disclosures.columns else None,
            str(pd.to_datetime(disclosures["published_at"], errors="coerce").max().date()) if not disclosures.empty and "published_at" in disclosures.columns else None,
        ],
        "price_date_range": [
            str(pd.to_datetime(prices["date"], errors="coerce").min().date()) if not prices.empty and "date" in prices.columns else None,
            str(pd.to_datetime(prices["date"], errors="coerce").max().date()) if not prices.empty and "date" in prices.columns else None,
        ],
        "run_duration_seconds": round(time.monotonic() - started_at, 3),
        "routine_status": routine_status.to_dict(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_backtest_report(out_dir: Path, curve: pd.DataFrame, simulation) -> None:
    if curve.empty:
        return
    working = curve.copy()
    working["date"] = pd.to_datetime(working["date"])
    working["year"] = working["date"].dt.year
    yearly = (
        working.groupby("year")
        .agg(
            start_value=("portfolio_value", "first"),
            ending_value=("portfolio_value", "last"),
            contribution=("strategy_contribution", "sum"),
            benchmark_ending_value=("benchmark_value", "last"),
        )
        .reset_index()
    )
    yearly["return"] = yearly["ending_value"] / yearly["start_value"] - 1.0
    yearly.to_csv(out_dir / "backtest_yearly_returns.csv", index=False)
    trades = pd.DataFrame([trade.__dict__ for trade in simulation.trades])
    if not trades.empty:
        trades.to_csv(out_dir / "backtest_trades.csv", index=False)
    max_drawdown = float(simulation.metrics.get("max_drawdown", 0.0))
    turnover = float(trades["gross_notional"].sum() / working["portfolio_value"].mean()) if not trades.empty else 0.0
    average_holding_period = 0.0
    monthly_contributions = working.loc[working["strategy_contribution"] > 0, ["date", "strategy_contribution"]]
    report = [
        "# Backtest Scenario Report",
        "",
        f"- Ending value: {simulation.metrics.get('ending_value', 0.0):.2f}",
        f"- Max drawdown: {max_drawdown:.2%}",
        f"- Turnover: {turnover:.2f}x",
        f"- Average holding period: {average_holding_period:.1f} days",
        f"- Trade count: {len(simulation.trades)}",
        f"- Monthly contribution events: {len(monthly_contributions)}",
    ]
    if not trades.empty:
        report.extend(
            [
                f"- Best trade notional: {trades['gross_notional'].max():.2f}",
                f"- Worst trade notional: {trades['gross_notional'].min():.2f}",
            ]
        )
    (out_dir / "backtest_report.md").write_text("\n".join(report) + "\n")


def _write_benchmark_report(out_dir: Path, prices: pd.DataFrame, benchmark_symbols: list[str], initial_cash: float) -> None:
    if prices.empty or not benchmark_symbols:
        return
    rows: list[dict[str, object]] = []
    normalized = prices.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    for symbol in benchmark_symbols:
        history = normalized.loc[normalized["symbol"] == symbol, ["date", "close"]].dropna().sort_values("date")
        if history.empty:
            continue
        first = history.iloc[0]
        latest = history.iloc[-1]
        units = initial_cash / float(first["close"]) if float(first["close"]) > 0 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "start_date": first["date"],
                "latest_date": latest["date"],
                "start_price": float(first["close"]),
                "latest_price": float(latest["close"]),
                "initial_cash_value": units * float(latest["close"]),
                "simple_return": (float(latest["close"]) / float(first["close"])) - 1.0 if float(first["close"]) > 0 else 0.0,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "benchmark_report.csv", index=False)


async def _run(config_path: str, command: str, dry_run: bool = False) -> None:
    started_at = time.monotonic()
    # One validated config object is shared across scraping, signal generation,
    # backtesting, and live paper execution.
    config = load_config(config_path)
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if command == "discord-test":
        sent = send_alert(
            "Capitol bot Discord test",
            "Discord webhook delivery is working for this container.",
            raise_on_error=True,
        )
        print(f"discord_alert_sent: {sent}")
        return

    if command == "daily-report":
        # Report-only mode renders the newest stored run from Postgres.
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL is required for daily-report")
        with connect_db(db_url) as conn:
            # Reporting against a fresh database should yield a clean "no run"
            # message path rather than failing on missing tables.
            ensure_schema(conn)
            report = load_latest_daily_report(conn)
            history = load_portfolio_history(conn)
        if report is None:
            raise RuntimeError("No paper run found. Run paper-run first.")
        message = render_daily_message(report)
        (out_dir / "daily_message.txt").write_text(message + "\n")
        history.to_csv(out_dir / "portfolio_history.csv", index=False)
        chart_path = out_dir / "portfolio_chart.svg"
        chart_path.write_text(render_portfolio_chart_svg(history))
        chart_png_path = out_dir / "portfolio_chart.png"
        write_portfolio_chart_png(history, str(chart_png_path))
        alert_sent = send_alert("Capitol bot daily update", message, attachment_path=chart_png_path)
        print(message)
        print(f"discord_alert_sent: {alert_sent}")
        return

    # The simplified strategy is disclosure driven, so the recent trade feed is
    # the authoritative signal input. A short-lived cache gives the daily
    # routine a controlled fallback when the public site has a transient error.
    scraper = CapitolTradesScraper(
        user_agent=config.data.user_agent,
        timeout_seconds=config.data.request_timeout_seconds,
    )
    disclosure_history_start = (
        pd.Timestamp(config.strategy.portfolio_start_date).normalize()
        - pd.Timedelta(days=config.strategy.signal_lookback_days - 1)
    ).date()
    trade_frames: list[pd.DataFrame] = []
    # Fetch the recent disclosure window used by the daily signal engine.
    recent_cache_path = cache_dir / f"recent_disclosures_{disclosure_history_start}.csv"
    try:
        recent_disclosures = scraper.to_frame(
            scraper.fetch_trades_since(
                min_published_date=disclosure_history_start,
                max_pages=config.data.trade_pages,
            )
        )
        if not recent_disclosures.empty:
            write_cached_frame(recent_disclosures, recent_cache_path)
    except Exception:
        recent_disclosures = (
            read_cached_frame(recent_cache_path, parse_dates=["traded_at", "published_at"])
            if cache_is_fresh(recent_cache_path, config.data.recent_disclosures_cache_hours)
            else pd.DataFrame()
        )
    if not recent_disclosures.empty:
        trade_frames.append(recent_disclosures)
    # The January seed basket is ranked from the configured prior year. Cache
    # that deeper historical scrape so the VPS does not need to refetch it on
    # every normal daily run after the initial seed has been established.
    seed_history_start = pd.Timestamp(f"{config.strategy.initial_selection_year}-01-01").date()
    if seed_history_start < disclosure_history_start:
        seed_cache_path = cache_dir / f"seed_disclosures_{config.strategy.initial_selection_year}.csv"
        if seed_cache_path.exists():
            seed_frame = pd.read_csv(seed_cache_path, parse_dates=["traded_at", "published_at"])
        else:
            seed_frame = scraper.to_frame(
                scraper.fetch_trades_since(
                    min_published_date=seed_history_start,
                    max_pages=config.data.seed_trade_pages,
                )
            )
            seed_frame.to_csv(seed_cache_path, index=False)
        if not seed_frame.empty:
            trade_frames.append(seed_frame)
    if trade_frames:
        trade_disclosures = (
            pd.concat(trade_frames, ignore_index=True)
            .drop_duplicates(
                subset=[
                    "politician",
                    "ticker",
                    "asset_name",
                    "traded_at",
                    "published_at",
                    "transaction_type",
                    "notional_mid",
                ]
            )
            .reset_index(drop=True)
        )
    else:
        trade_disclosures = pd.DataFrame()

    benchmark_symbol = config.strategy.benchmark_symbol
    tradable_tickers = []
    if not trade_disclosures.empty and "ticker" in trade_disclosures.columns:
        trade_disclosures = trade_disclosures[trade_disclosures["ticker"].notna() & (trade_disclosures["ticker"] != "")]
        tradable_tickers = [str(ticker) for ticker in trade_disclosures["ticker"].dropna().tolist() if ticker]
    report_benchmarks = list(config.strategy.benchmark_report_symbols)
    symbols = sorted(set(tradable_tickers + [benchmark_symbol] + report_benchmarks)) if tradable_tickers else sorted(set([benchmark_symbol] + report_benchmarks))

    market = MarketDataClient(
        provider_priority=config.data.provider_priority,
        min_history_rows=config.data.min_history_rows,
    )
    # Resolve noisy Capitol symbols into provider-friendly market-data symbols.
    price_cache_path = cache_dir / f"prices_{pd.Timestamp(config.strategy.start_date).date()}_{pd.Timestamp.utcnow().date()}.csv"
    resolution_cache_path = cache_dir / f"symbol_resolution_{pd.Timestamp(config.strategy.start_date).date()}_{pd.Timestamp.utcnow().date()}.csv"
    try:
        prices, symbol_resolution = await market.resolve_and_fetch_history(
            symbols,
            config.strategy.start_date,
            config.strategy.end_date,
        )
        if not prices.empty:
            write_cached_frame(prices, price_cache_path)
            write_cached_frame(symbol_resolution, resolution_cache_path)
    except Exception:
        if cache_is_fresh(price_cache_path, config.data.price_cache_hours):
            prices = read_cached_frame(price_cache_path, parse_dates=["date"])
            symbol_resolution = read_cached_frame(resolution_cache_path)
        else:
            raise
    resolved_symbols = (
        set(symbol_resolution.loc[symbol_resolution["status"] == "ok", "raw_symbol"].tolist())
        if not symbol_resolution.empty
        else set()
    )
    if resolved_symbols and not trade_disclosures.empty and "ticker" in trade_disclosures.columns:
        trade_disclosures = trade_disclosures[trade_disclosures["ticker"].isin(resolved_symbols)].copy()

    strategy = CapitolStrategy(config)
    latest_signal_frame = strategy.build_signal_frame(trade_disclosures, prices)
    latest_targets = strategy.build_targets(latest_signal_frame, current_positions=None, available_cash=config.execution.initial_cash)
    routine_status = build_routine_status(
        trade_disclosures,
        prices,
        symbol_resolution,
        benchmark_symbol,
        config,
    )

    # Persist flat files for audit/debugging after each run.
    trade_disclosures.to_csv(out_dir / "disclosures.csv", index=False)
    prices.to_csv(out_dir / "prices.csv", index=False)
    symbol_resolution.to_csv(out_dir / "symbol_resolution.csv", index=False)
    latest_signal_frame.to_csv(out_dir / "signal_frame.csv", index=False)
    latest_targets.to_csv(out_dir / "targets.csv", index=False)
    _write_benchmark_report(
        out_dir,
        prices,
        [benchmark_symbol] + list(config.strategy.benchmark_report_symbols),
        float(config.execution.initial_cash),
    )
    (out_dir / "routine_status.json").write_text(json.dumps(routine_status.to_dict(), indent=2, sort_keys=True) + "\n")
    _write_manifest(out_dir, config_path, command, started_at, routine_status, trade_disclosures, prices, "routine-built")
    if routine_status.provider_error_count:
        send_alert("Capitol bot provider error", routine_status.reason)
    if routine_status.stale_disclosures or routine_status.disclosure_count == 0:
        send_alert("Capitol bot disclosure feed warning", routine_status.reason)

    if command == "health-check":
        if routine_status.degraded:
            send_alert("Capitol bot health degraded", routine_status.reason)
        print(json.dumps(routine_status.to_dict(), indent=2, sort_keys=True))
        return

    if command == "backtest":
        # Walk forward day by day using only information available on each date.
        simulation = strategy.simulate_walk_forward(trade_disclosures, prices, benchmark_symbol)
        curve, metrics = simulation.curve, simulation.metrics
        curve.to_csv(out_dir / "equity_curve.csv", index=False)
        _write_backtest_report(out_dir, curve, simulation)
        _write_manifest(out_dir, config_path, command, started_at, routine_status, trade_disclosures, prices, "ok")
        print(format_metrics(metrics))
        if not latest_targets.empty:
            print("\nTop targets")
            print(latest_targets.sort_values("weight", ascending=False).to_string(index=False))
        return

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required for paper-run")
    if routine_status.stale_price:
        send_alert("Capitol bot stale data", routine_status.reason)
        raise RuntimeError(f"Stale data, no trade: {routine_status.reason}")
    with connect_db(db_url) as conn:
        paper_trader = PaperTrader(conn=conn, config=config, apply_schema=not dry_run)
        paper_result = paper_trader.run(
            disclosures=trade_disclosures,
            prices=prices,
            symbol_resolution=symbol_resolution,
            strategy=strategy,
            benchmark_symbol=benchmark_symbol,
            command=command,
            run_metadata=routine_status.to_dict(),
            dry_run=dry_run,
        )
        report = None if dry_run else load_latest_daily_report(conn)
        history = load_portfolio_history(conn)

    # Use the persisted daily report as the source of truth for the text output.
    if report is not None:
        message = render_daily_message(report)
        (out_dir / "daily_message.txt").write_text(message + "\n")

    # Persist the exact signal/target view associated with this run when it is available.
    run_signal_frame = paper_result.signal_frame if not paper_result.signal_frame.empty else latest_signal_frame
    run_targets = paper_result.targets if not paper_result.targets.empty else latest_targets
    run_signal_frame.to_csv(out_dir / "signal_frame.csv", index=False)
    run_targets.to_csv(out_dir / "targets.csv", index=False)
    paper_result.target_diff.to_csv(out_dir / "target_diff.csv", index=False)
    history.to_csv(out_dir / "portfolio_history.csv", index=False)
    chart_path = out_dir / "portfolio_chart.svg"
    chart_path.write_text(render_portfolio_chart_svg(history))
    chart_png_path = out_dir / "portfolio_chart.png"
    write_portfolio_chart_png(history, str(chart_png_path))
    if report is not None and not paper_result.already_ran_today:
        send_alert("Capitol bot daily update", message, attachment_path=chart_png_path)
    _write_manifest(out_dir, config_path, command, started_at, routine_status, trade_disclosures, prices, "dry-run" if dry_run else "ok")

    action_summary = paper_result.action_summary
    action_reason = paper_result.action_reason
    if paper_result.already_ran_today:
        # A same-day rerun should read as an idempotent no-op rather than as a
        # second execution of the already-persisted trades and contributions.
        action_summary = "No-op (reused existing run for today)"
        action_reason = "duplicate same-day invocation; no new contributions or trades were executed"
    print(
        "\n".join(
            [
                f"paper_run_id: {paper_result.run_id}",
                f"dry_run: {dry_run}",
                f"run_date: {paper_result.run_date}",
                f"already_ran_today: {paper_result.already_ran_today}",
                f"contribution_applied: {paper_result.contribution_applied}",
                f"strategy_contribution: {paper_result.strategy_contribution_amount:.2f}",
                f"benchmark_contribution: {paper_result.benchmark_contribution_amount:.2f}",
                f"cash: {paper_result.cash:.2f}",
                f"market_value: {paper_result.market_value:.2f}",
                f"portfolio_value: {paper_result.portfolio_value:.2f}",
                f"benchmark_value: {paper_result.benchmark_value:.2f}",
                f"trade_count: {len(paper_result.trades)}",
                f"routine_status: {routine_status.reason}",
                f"action: {action_summary}",
                f"reason: {action_reason}",
            ]
        )
    )
    if report is not None:
        heading = "Daily message (stored run result)" if paper_result.already_ran_today else "Daily message"
        print(f"\n{heading}")
        print(message)
    if paper_result.trades:
        send_alert("Capitol bot trade executed", action_summary)
        if paper_result.already_ran_today:
            print("\nStored trades from earlier successful run")
        else:
            print("\nTrades")
        for trade in paper_result.trades:
            print(
                f"{trade.side.upper()} {trade.symbol} qty={trade.quantity:.4f} price={trade.price:.2f} "
                f"notional={trade.gross_notional:.2f} fees={trade.fees:.2f} reason={trade.reason}"
            )


def main() -> None:
    # The async wrapper is needed because market-data fetches use worker threads.
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args.config, args.command, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
