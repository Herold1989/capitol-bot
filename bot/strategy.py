from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from bot.config import AppConfig
from bot.models import PositionTarget


@dataclass(slots=True)
class BacktestTrade:
    # Compact trade record used only inside the walk-forward simulator.
    date: pd.Timestamp
    symbol: str
    side: str
    gross_notional: float


@dataclass(slots=True)
class StrategyArtifacts:
    # Convenience bundle for consumers that want the latest signal frame and the
    # resulting target book together.
    signal_frame: pd.DataFrame
    targets: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


@dataclass(slots=True)
class SimulationState:
    # Full walk-forward state used by both the backtest command and the
    # first-run live seeding path.
    cash: float
    positions: dict[str, dict[str, float]]
    benchmark_units: float
    total_contributed_strategy: float
    total_contributed_benchmark: float
    last_strategy_contribution_date: date | None
    last_benchmark_contribution_date: date | None
    trades: list[BacktestTrade]
    curve: pd.DataFrame
    metrics: dict[str, float]


def _normalize_price_history(prices: pd.DataFrame) -> pd.DataFrame:
    # Strategy logic relies on normalized daily timestamps and numeric closes.
    if prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    columns = ["date", "symbol", "close"] + (["volume"] if "volume" in prices.columns else [])
    frame = prices.loc[:, columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame


def _normalize_disclosures(disclosures: pd.DataFrame) -> pd.DataFrame:
    # Only a small subset of disclosure columns is required for this simpler model.
    if disclosures.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "asset_name",
                "politician",
                "traded_at",
                "published_at",
                "transaction_type",
                "signed_notional",
                "notional_mid",
            ]
        )
    frame = disclosures.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"]).dt.normalize()
    if "traded_at" in frame.columns:
        frame["traded_at"] = pd.to_datetime(frame["traded_at"], errors="coerce").dt.normalize()
    else:
        frame["traded_at"] = frame["published_at"]
    frame["signed_notional"] = pd.to_numeric(frame["signed_notional"], errors="coerce")
    if "notional_mid" in frame.columns:
        frame["notional_mid"] = pd.to_numeric(frame["notional_mid"], errors="coerce")
    else:
        frame["notional_mid"] = frame["signed_notional"].abs()
    frame["transaction_type"] = frame["transaction_type"].astype(str).str.lower()
    frame = frame.dropna(subset=["ticker", "published_at", "signed_notional"])
    frame = frame[frame["ticker"].astype(str) != ""].copy()
    return frame.reset_index(drop=True)


def _politician_reliability(disclosures: pd.DataFrame, prices: pd.DataFrame, lookahead_days: int) -> pd.DataFrame:
    if disclosures.empty or prices.empty or "politician" not in disclosures.columns:
        return pd.DataFrame(columns=["politician", "avg_forward_return", "hit_rate", "filing_delay_days"])
    price_pivot = prices.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
    rows: list[dict[str, object]] = []
    for _, row in disclosures.dropna(subset=["politician", "ticker", "published_at"]).iterrows():
        symbol = str(row["ticker"])
        if symbol not in price_pivot.columns:
            continue
        series = price_pivot[symbol].dropna()
        eligible = series[series.index >= row["published_at"]]
        if eligible.empty:
            continue
        start_idx = eligible.index[0]
        future = series[series.index > start_idx].head(max(int(lookahead_days), 1))
        if future.empty or float(series.loc[start_idx]) <= 0:
            continue
        forward_return = (float(future.iloc[-1]) / float(series.loc[start_idx])) - 1.0
        direction = 1.0 if str(row["transaction_type"]).startswith("buy") else -1.0
        traded_at = row.get("traded_at", row["published_at"])
        delay_days = (row["published_at"] - traded_at).days if pd.notna(traded_at) else 0
        rows.append(
            {
                "politician": row["politician"],
                "forward_return": forward_return * direction,
                "hit": (forward_return * direction) > 0,
                "filing_delay_days": max(int(delay_days), 0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["politician", "avg_forward_return", "hit_rate", "filing_delay_days"])
    return (
        pd.DataFrame(rows)
        .groupby("politician", dropna=True)
        .agg(
            avg_forward_return=("forward_return", "mean"),
            hit_rate=("hit", "mean"),
            filing_delay_days=("filing_delay_days", "mean"),
        )
        .reset_index()
    )


def _first_available_price_on_or_after(price_series: pd.Series, start_ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    # Benchmark initialization should use the first available trading close on
    # or after the configured benchmark start date.
    eligible = price_series[price_series.index >= start_ts]
    if eligible.empty:
        return None
    first_idx = eligible.index[0]
    return first_idx, float(eligible.iloc[0])


def _slot_weight(config: AppConfig, active_count: int) -> float:
    # When cash buffering is enabled, each active name receives one equal slot
    # out of the maximum number of positions, leaving unused slots in cash.
    if active_count <= 0:
        return 0.0
    if config.strategy.allow_cash_buffer:
        return 1.0 / float(config.strategy.max_positions)
    return 1.0 / float(active_count)


def _cap_slot_weight(config: AppConfig, active_count: int) -> float:
    return min(_slot_weight(config, active_count), float(config.strategy.max_single_position_weight))


def _normalize_current_symbols(current_positions: pd.DataFrame | dict[str, dict[str, float]] | Iterable[str] | None) -> set[str]:
    # build_targets accepts lightweight current-position inputs so both the live
    # trader and the backtest can call it without reshaping their state first.
    if current_positions is None:
        return set()
    if isinstance(current_positions, dict):
        return {symbol for symbol, state in current_positions.items() if float(state.get("quantity", 0.0)) > 1e-8}
    if isinstance(current_positions, pd.DataFrame):
        if current_positions.empty or "symbol" not in current_positions.columns:
            return set()
        if "quantity" in current_positions.columns:
            return {
                str(row["symbol"])
                for _, row in current_positions.iterrows()
                if float(row.get("quantity", 0.0) or 0.0) > 1e-8
            }
        return {str(symbol) for symbol in current_positions["symbol"].dropna().tolist()}
    return {str(symbol) for symbol in current_positions}


def _is_contribution_due(run_date: date, last_contribution_date: date | None, contribution_day: int) -> bool:
    # Contributions should happen once per month on the first eligible run on or
    # after contribution_day, which naturally becomes the first trading day in
    # backtests built from trading-date price series.
    if run_date.day < contribution_day:
        return False
    if last_contribution_date is None:
        return True
    return (last_contribution_date.year, last_contribution_date.month) != (run_date.year, run_date.month)


class CapitolStrategy:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_signal_frame(
        self,
        disclosures: pd.DataFrame,
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp | date | str | None = None,
    ) -> pd.DataFrame:
        # Signal construction is deliberately simple: only recent published
        # disclosures plus a 50-day moving-average trend check.
        normalized_disclosures = _normalize_disclosures(disclosures)
        normalized_prices = _normalize_price_history(prices)
        reliability = _politician_reliability(
            normalized_disclosures,
            normalized_prices,
            self.config.strategy.reliability_lookahead_days,
        )
        if normalized_prices.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "asset_name",
                    "published_count",
                    "buy_count",
                    "sell_count",
                    "net_notional",
                    "recency_weighted_net_notional",
                    "latest_close",
                    "moving_average",
                    "above_moving_average",
                    "score",
                    "passes_buy_filters",
                    "rank",
                    "is_buyable",
                    "buy_reason",
                    "sell_reason",
                    "signal_date",
                ]
            )

        signal_ts = pd.Timestamp(as_of_date or normalized_prices["date"].max()).normalize()
        # Price filters only use history available up to the signal date.
        price_window = normalized_prices[normalized_prices["date"] <= signal_ts].copy()
        price_window["moving_average"] = (
            price_window.groupby("symbol")["close"]
            .transform(lambda series: series.rolling(self.config.strategy.moving_average_days).mean())
        )
        if "volume" in price_window.columns:
            price_window["avg_dollar_volume"] = (
                (price_window["close"] * price_window["volume"])
                .groupby(price_window["symbol"])
                .transform(lambda series: series.rolling(self.config.strategy.moving_average_days).mean())
            )
        latest_price_rows = price_window.groupby("symbol").tail(1).set_index("symbol")
        feature_columns = ["close", "moving_average"] + (["avg_dollar_volume"] if "avg_dollar_volume" in latest_price_rows.columns else [])
        latest_features = latest_price_rows.loc[:, feature_columns].rename(columns={"close": "latest_close"})

        lookback_start = signal_ts - pd.Timedelta(days=self.config.strategy.signal_lookback_days - 1)
        aggregated = pd.DataFrame(
            columns=[
                "asset_name",
                "published_count",
                "buy_count",
                "sell_count",
                "net_notional",
                "recency_weighted_net_notional",
                "delay_weighted_net_notional",
                "latest_published_at",
            ]
        )
        if not normalized_disclosures.empty:
            # The strategy must use published_at only, never traded_at.
            window = normalized_disclosures[
                (normalized_disclosures["published_at"] <= signal_ts)
                & (normalized_disclosures["published_at"] >= lookback_start)
            ].copy()
            if not window.empty:
                # Newer disclosures matter more than older ones, but the weighting stays
                # intentionally transparent and bounded.
                window["age_days"] = (signal_ts - window["published_at"]).dt.days.clip(lower=0)
                window["recency_weight"] = (
                    (self.config.strategy.signal_lookback_days - window["age_days"]).clip(lower=0) + 1
                ) / float(self.config.strategy.signal_lookback_days)
                window["recency_weighted_notional"] = window["signed_notional"] * window["recency_weight"]
                window["is_buy"] = window["transaction_type"].str.startswith("buy")
                window["is_sell"] = window["transaction_type"].str.startswith("sell")
                window["filing_delay_days"] = (window["published_at"] - window["traded_at"]).dt.days.clip(lower=0)
                window["delay_weight"] = np.exp(
                    -window["filing_delay_days"] / float(self.config.strategy.disclosure_delay_half_life_days)
                )
                window["delay_weighted_notional"] = window["recency_weighted_notional"] * window["delay_weight"]
                if not reliability.empty and "politician" in window.columns:
                    reliability_scores = reliability.loc[:, ["politician", "avg_forward_return", "hit_rate"]]
                    window = window.merge(reliability_scores, on="politician", how="left")
                for column, default in {"avg_forward_return": 0.0, "hit_rate": 0.5}.items():
                    if column not in window.columns:
                        window[column] = default
                    window[column] = window[column].fillna(default)

                aggregated = (
                    window.groupby("ticker", dropna=True)
                    .agg(
                        asset_name=("asset_name", "first"),
                        published_count=("ticker", "size"),
                        buy_count=("is_buy", "sum"),
                        sell_count=("is_sell", "sum"),
                        net_notional=("signed_notional", "sum"),
                        recency_weighted_net_notional=("recency_weighted_notional", "sum"),
                        delay_weighted_net_notional=("delay_weighted_notional", "sum"),
                        latest_published_at=("published_at", "max"),
                        unique_politicians=("politician", "nunique") if "politician" in window.columns else ("ticker", "size"),
                        avg_filing_delay_days=("filing_delay_days", "mean"),
                        politician_avg_forward_return=("avg_forward_return", "mean"),
                        politician_hit_rate=("hit_rate", "mean"),
                    )
                    .rename_axis("symbol")
                )
            latest_history = (
                normalized_disclosures[normalized_disclosures["published_at"] <= signal_ts]
                .groupby("ticker", dropna=True)["published_at"]
                .max()
            )
            if not latest_history.empty:
                if "latest_published_at" not in aggregated.columns:
                    aggregated["latest_published_at"] = pd.NaT
                aggregated["latest_known_published_at"] = latest_history

        # Keep price-feature rows even when a held symbol has no fresh
        # disclosures in the 45-day window, so the strategy can continue to
        # hold trend-positive names instead of forcing unnecessary cash.
        signal_frame = latest_features.copy() if aggregated.empty else latest_features.join(aggregated, how="left")
        if signal_frame.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "asset_name",
                    "published_count",
                    "buy_count",
                    "sell_count",
                    "net_notional",
                    "recency_weighted_net_notional",
                    "latest_close",
                    "moving_average",
                    "above_moving_average",
                    "score",
                    "passes_buy_filters",
                    "rank",
                    "is_buyable",
                    "buy_reason",
                    "sell_reason",
                    "signal_date",
                ]
            )

        fallback_asset_names = pd.Series(signal_frame.index.astype(str), index=signal_frame.index)
        for column, default in {
            "asset_name": fallback_asset_names,
            "published_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "net_notional": 0.0,
            "recency_weighted_net_notional": 0.0,
            "delay_weighted_net_notional": 0.0,
            "latest_published_at": pd.NaT,
            "latest_known_published_at": pd.NaT,
            "unique_politicians": 0,
            "avg_filing_delay_days": 0.0,
            "politician_avg_forward_return": 0.0,
            "politician_hit_rate": 0.5,
            "avg_dollar_volume": np.nan,
        }.items():
            if column not in signal_frame.columns:
                signal_frame[column] = default

        signal_frame["is_leveraged_or_inverse_etf"] = signal_frame["asset_name"].astype(str).str.contains(
            r"\b(?:2x|3x|leveraged|inverse|ultra|bear|short)\b",
            case=False,
            regex=True,
        )
        signal_frame["asset_name"] = signal_frame["asset_name"].fillna(fallback_asset_names)
        signal_frame["published_count"] = signal_frame["published_count"].fillna(0).astype(int)
        signal_frame["buy_count"] = signal_frame["buy_count"].fillna(0).astype(int)
        signal_frame["sell_count"] = signal_frame["sell_count"].fillna(0).astype(int)
        signal_frame["net_notional"] = signal_frame["net_notional"].fillna(0.0)
        signal_frame["recency_weighted_net_notional"] = signal_frame["recency_weighted_net_notional"].fillna(0.0)
        signal_frame["delay_weighted_net_notional"] = signal_frame["delay_weighted_net_notional"].fillna(0.0)
        signal_frame["unique_politicians"] = signal_frame["unique_politicians"].fillna(0).astype(int)
        signal_frame["avg_filing_delay_days"] = signal_frame["avg_filing_delay_days"].fillna(0.0)
        signal_frame["politician_avg_forward_return"] = signal_frame["politician_avg_forward_return"].fillna(0.0)
        signal_frame["politician_hit_rate"] = signal_frame["politician_hit_rate"].fillna(0.5)
        signal_frame["avg_dollar_volume"] = pd.to_numeric(signal_frame["avg_dollar_volume"], errors="coerce")
        latest_published = pd.to_datetime(
            signal_frame["latest_published_at"].fillna(signal_frame["latest_known_published_at"]),
            errors="coerce",
        )
        signal_frame["days_since_latest_disclosure"] = (signal_ts - latest_published).dt.days
        signal_frame["above_moving_average"] = signal_frame["latest_close"] > signal_frame["moving_average"]
        scaled_notional = np.sign(signal_frame["delay_weighted_net_notional"]) * np.log1p(
            signal_frame["delay_weighted_net_notional"].abs()
        )
        signal_frame["buy_score"] = (
            scaled_notional
            + 0.25 * signal_frame["buy_count"]
            - 0.25 * signal_frame["sell_count"]
            + 0.10 * signal_frame["unique_politicians"]
            + 2.0 * signal_frame["politician_avg_forward_return"]
            + 0.50 * (signal_frame["politician_hit_rate"] - 0.5)
        )
        signal_frame["sell_risk"] = (
            0.50 * signal_frame["sell_count"]
            + np.where(signal_frame["net_notional"] < 0, 2.0, 0.0)
            + np.where(~signal_frame["above_moving_average"].fillna(False), 2.0, 0.0)
            + (signal_frame["avg_filing_delay_days"] / 60.0).clip(upper=2.0)
        )
        signal_frame["score"] = signal_frame["buy_score"] - signal_frame["sell_risk"]

        liquidity_ok = signal_frame["avg_dollar_volume"].isna() | (
            signal_frame["avg_dollar_volume"] >= float(self.config.strategy.min_avg_dollar_volume)
        )
        signal_frame["passes_buy_filters"] = (
            (signal_frame["net_notional"] > 0)
            & (signal_frame["buy_count"] >= self.config.strategy.min_buy_disclosures)
            & signal_frame["above_moving_average"].fillna(False)
            & liquidity_ok
            & (~signal_frame["is_leveraged_or_inverse_etf"])
        )
        signal_frame["rank"] = np.nan
        ranked = signal_frame[signal_frame["passes_buy_filters"]].sort_values(
            ["score", "buy_count", "net_notional", "latest_published_at"],
            ascending=[False, False, False, False],
        )
        if not ranked.empty:
            signal_frame.loc[ranked.index, "rank"] = range(1, len(ranked) + 1)
        signal_frame["is_buyable"] = signal_frame["passes_buy_filters"] & (signal_frame["rank"] <= self.config.strategy.max_positions)
        signal_frame["hard_sell"] = (signal_frame["net_notional"] < 0) | (~signal_frame["above_moving_average"].fillna(False))

        signal_frame["buy_reason"] = np.where(
            signal_frame["is_buyable"],
            "positive 45d net Capitol flow, >=2 buys, and close above 50d MA",
            "",
        )
        signal_frame["sell_reason"] = np.where(
            signal_frame["net_notional"] < 0,
            "net disclosure flow turned negative",
            np.where(
                ~signal_frame["above_moving_average"].fillna(False),
                "latest close fell below 50d MA",
                np.where(
                    signal_frame["is_leveraged_or_inverse_etf"],
                    "leveraged or inverse ETF skipped",
                    np.where(
                    signal_frame["published_count"] <= 0,
                    "no fresh Capitol disclosures; retain only if no stronger candidate competes",
                    "ranked below stronger candidate set",
                ),
                ),
            ),
        )
        signal_frame["signal_date"] = signal_ts

        return (
            signal_frame.reset_index()
            .sort_values(["is_buyable", "score", "buy_count", "net_notional"], ascending=[False, False, False, False])
            .reset_index(drop=True)
        )

    def build_targets(
        self,
        signal_frame: pd.DataFrame,
        current_positions: pd.DataFrame | dict[str, dict[str, float]] | Iterable[str] | None,
        available_cash: float,
        as_of_date: pd.Timestamp | date | str | None = None,
    ) -> pd.DataFrame:
        # Targets are just the active top-3 slot book; execution decides whether
        # a rebalance is actually necessary today.
        del available_cash
        current_symbols = _normalize_current_symbols(current_positions)
        if signal_frame.empty:
            return pd.DataFrame(columns=["date", "symbol", "weight", "score", "reason"])

        working = signal_frame.copy()
        working["is_current_hold"] = working["symbol"].isin(current_symbols)
        if isinstance(current_positions, dict):
            avg_costs = {symbol: float(state.get("avg_cost", 0.0) or 0.0) for symbol, state in current_positions.items()}
            working["current_avg_cost"] = working["symbol"].map(avg_costs).fillna(0.0)
            working["stop_loss_triggered"] = (
                working["is_current_hold"]
                & (working["current_avg_cost"] > 0)
                & (working["latest_close"] <= working["current_avg_cost"] * (1.0 - float(self.config.strategy.stop_loss_pct)))
            )
        else:
            working["stop_loss_triggered"] = False
        disclosure_age = pd.to_numeric(working.get("days_since_latest_disclosure"), errors="coerce")
        working["hold_is_too_stale"] = disclosure_age.isna() | (disclosure_age > int(self.config.strategy.max_stale_hold_days))
        working["candidate_priority"] = np.where(
            working["is_buyable"],
            2,
            np.where(
                working["is_current_hold"]
                & ~working["hard_sell"]
                & ~working["hold_is_too_stale"]
                & ~working["stop_loss_triggered"],
                1,
                0,
            ),
        )
        active = working[working["candidate_priority"] > 0].copy()
        if active.empty:
            return pd.DataFrame(columns=["date", "symbol", "weight", "score", "reason"])

        active = active.sort_values(
            ["candidate_priority", "is_buyable", "rank", "score", "buy_count", "net_notional"],
            ascending=[False, False, True, False, False, False],
        ).head(self.config.strategy.max_positions)
        dropped_current = working[
            working["is_current_hold"]
            & ~working["hard_sell"]
            & ~working["hold_is_too_stale"]
            & ~working["stop_loss_triggered"]
            & ~working["symbol"].isin(active["symbol"])
        ].sort_values("score", ascending=False)
        if not dropped_current.empty:
            for _, hold_row in dropped_current.iterrows():
                new_active = active[~active["symbol"].isin(current_symbols)].sort_values("score", ascending=True)
                if new_active.empty:
                    break
                weakest_new = new_active.iloc[0]
                if float(weakest_new["score"]) - float(hold_row["score"]) >= float(self.config.strategy.min_replacement_score_advantage):
                    continue
                active = pd.concat(
                    [active[active["symbol"] != weakest_new["symbol"]], hold_row.to_frame().T],
                    ignore_index=True,
                )
        slot_weight = _cap_slot_weight(self.config, len(active))
        target_date = pd.Timestamp(as_of_date or active["signal_date"].max()).normalize()

        targets: list[PositionTarget] = []
        rows: list[dict[str, object]] = []
        for _, row in active.iterrows():
            targets.append(PositionTarget(symbol=str(row["symbol"]), weight=float(slot_weight), score=float(row["score"])))
            if row["symbol"] in current_symbols and not bool(row["is_buyable"]):
                reason = "hold existing position while trend remains intact and no stronger candidate displaced it"
            else:
                reason = str(row["buy_reason"] or "")
            if row["symbol"] in current_symbols and bool(row["is_buyable"]):
                reason = f"hold target: {reason}" if reason else "hold target"
            elif row["symbol"] in current_symbols:
                reason = f"hold target: {reason}" if reason else "hold target"
            else:
                reason = f"buy candidate: {reason}" if reason else "buy candidate"
            rows.append(
                asdict(targets[-1])
                | {
                    "date": target_date,
                    "reason": reason,
                }
            )
        return pd.DataFrame(rows, columns=["date", "symbol", "weight", "score", "sector", "reason"]).drop(columns=["sector"])

    def build_initial_seed_targets(
        self,
        disclosures: pd.DataFrame,
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp | date | str,
    ) -> pd.DataFrame:
        # Seed the strategy with the most-bought Capitol names from the prior
        # year so the January portfolio begins invested instead of in cash.
        normalized_disclosures = _normalize_disclosures(disclosures)
        normalized_prices = _normalize_price_history(prices)
        if normalized_disclosures.empty or normalized_prices.empty:
            return pd.DataFrame(columns=["date", "symbol", "weight", "score", "reason"])

        selection_year = int(self.config.strategy.initial_selection_year)
        seed_ts = pd.Timestamp(as_of_date).normalize()
        year_window = normalized_disclosures[
            (normalized_disclosures["published_at"] >= pd.Timestamp(f"{selection_year}-01-01"))
            & (normalized_disclosures["published_at"] <= pd.Timestamp(f"{selection_year}-12-31"))
            & (normalized_disclosures["transaction_type"].str.startswith("buy"))
        ].copy()
        if year_window.empty:
            return pd.DataFrame(columns=["date", "symbol", "weight", "score", "reason"])

        ranked = (
            year_window.groupby("ticker", dropna=True)
            .agg(
                asset_name=("asset_name", "first"),
                buy_count=("ticker", "size"),
                net_notional=("signed_notional", "sum"),
            )
            .rename_axis("symbol")
            .sort_values(["buy_count", "net_notional"], ascending=[False, False])
            .head(self.config.strategy.initial_top_n)
            .reset_index()
        )
        available_symbols = set(
            normalized_prices.loc[normalized_prices["date"] <= seed_ts, "symbol"].dropna().astype(str).tolist()
        )
        ranked = ranked[ranked["symbol"].isin(available_symbols)].copy()
        if ranked.empty:
            return pd.DataFrame(columns=["date", "symbol", "weight", "score", "reason"])

        slot_weight = min(1.0 / float(len(ranked)), float(self.config.strategy.max_single_position_weight))
        ranked["date"] = seed_ts
        ranked["weight"] = slot_weight
        ranked["score"] = ranked["buy_count"].astype(float) + ranked["net_notional"].astype(float) / 1_000_000.0
        ranked["reason"] = f"initial seed basket from top {self.config.strategy.initial_top_n} Capitol buy names in {selection_year}"
        return ranked.loc[:, ["date", "symbol", "weight", "score", "reason"]]

    def backtest_walk_forward(
        self,
        disclosures: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_symbol: str,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        # The public backtest command only needs the curve and metrics, while
        # the shared simulator also exposes final positions for live seeding.
        simulation = self.simulate_walk_forward(
            disclosures=disclosures,
            prices=prices,
            benchmark_symbol=benchmark_symbol,
        )
        return simulation.curve, simulation.metrics

    def simulate_walk_forward(
        self,
        disclosures: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_symbol: str,
        end_date: pd.Timestamp | date | str | None = None,
    ) -> SimulationState:
        # The walk-forward simulator mirrors the live bot: initial funding once,
        # monthly contributions once per month, signals from published_at only,
        # and trades only when the target set or new cash requires action.
        normalized_prices = _normalize_price_history(prices)
        normalized_disclosures = _normalize_disclosures(disclosures)
        if normalized_prices.empty:
            empty_curve = pd.DataFrame(
                columns=[
                    "date",
                    "cash",
                    "market_value",
                    "portfolio_value",
                    "benchmark_value",
                    "strategy_contribution",
                    "benchmark_contribution",
                    "total_contributed_strategy",
                    "total_contributed_benchmark",
                    "trade_count",
                ]
            )
            return SimulationState(
                cash=0.0,
                positions={},
                benchmark_units=0.0,
                total_contributed_strategy=0.0,
                total_contributed_benchmark=0.0,
                last_strategy_contribution_date=None,
                last_benchmark_contribution_date=None,
                trades=[],
                curve=empty_curve,
                metrics=self._empty_metrics(),
            )

        pivot = normalized_prices.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
        if benchmark_symbol not in pivot.columns:
            empty_curve = pd.DataFrame(
                columns=[
                    "date",
                    "cash",
                    "market_value",
                    "portfolio_value",
                    "benchmark_value",
                    "strategy_contribution",
                    "benchmark_contribution",
                    "total_contributed_strategy",
                    "total_contributed_benchmark",
                    "trade_count",
                ]
            )
            return SimulationState(
                cash=0.0,
                positions={},
                benchmark_units=0.0,
                total_contributed_strategy=0.0,
                total_contributed_benchmark=0.0,
                last_strategy_contribution_date=None,
                last_benchmark_contribution_date=None,
                trades=[],
                curve=empty_curve,
                metrics=self._empty_metrics(),
            )

        if self.config.strategy.end_date is None:
            configured_end_ts = pivot.index.max()
        else:
            configured_end_ts = min(pd.Timestamp(self.config.strategy.end_date).normalize(), pivot.index.max())
        effective_end_ts = configured_end_ts
        if end_date is not None:
            effective_end_ts = min(effective_end_ts, pd.Timestamp(end_date).normalize())

        portfolio_start_ts = pd.Timestamp(self.config.strategy.portfolio_start_date).normalize()
        strategy_trading_dates = [ts for ts in pivot.index if portfolio_start_ts <= ts <= effective_end_ts]
        if not strategy_trading_dates:
            empty_curve = pd.DataFrame(
                columns=[
                    "date",
                    "cash",
                    "market_value",
                    "portfolio_value",
                    "benchmark_value",
                    "strategy_contribution",
                    "benchmark_contribution",
                    "total_contributed_strategy",
                    "total_contributed_benchmark",
                    "trade_count",
                ]
            )
            return SimulationState(
                cash=0.0,
                positions={},
                benchmark_units=0.0,
                total_contributed_strategy=0.0,
                total_contributed_benchmark=0.0,
                last_strategy_contribution_date=None,
                last_benchmark_contribution_date=None,
                trades=[],
                curve=empty_curve,
                metrics=self._empty_metrics(),
            )

        benchmark_start_ts = pd.Timestamp(self.config.strategy.benchmark_start_date).normalize()
        benchmark_start_info = _first_available_price_on_or_after(pivot[benchmark_symbol].dropna(), benchmark_start_ts)
        if benchmark_start_info is None:
            empty_curve = pd.DataFrame(
                columns=[
                    "date",
                    "cash",
                    "market_value",
                    "portfolio_value",
                    "benchmark_value",
                    "strategy_contribution",
                    "benchmark_contribution",
                    "total_contributed_strategy",
                    "total_contributed_benchmark",
                    "trade_count",
                ]
            )
            return SimulationState(
                cash=0.0,
                positions={},
                benchmark_units=0.0,
                total_contributed_strategy=0.0,
                total_contributed_benchmark=0.0,
                last_strategy_contribution_date=None,
                last_benchmark_contribution_date=None,
                trades=[],
                curve=empty_curve,
                metrics=self._empty_metrics(),
            )
        benchmark_first_day, first_benchmark_price = benchmark_start_info
        first_strategy_day = strategy_trading_dates[0]
        first_simulation_day = min(first_strategy_day, benchmark_first_day)
        trading_dates = [ts for ts in pivot.index if first_simulation_day <= ts <= effective_end_ts]

        # Initial funding is injected on the configured portfolio and benchmark
        # start dates, not at the beginning of the longer indicator-history window.
        cash = 0.0
        total_contributed_strategy = 0.0
        total_contributed_benchmark = 0.0
        benchmark_units = 0.0
        last_strategy_contribution_date: date | None = None
        last_benchmark_contribution_date: date | None = None

        positions: dict[str, dict[str, float]] = {}
        trades: list[BacktestTrade] = []
        curve_rows: list[dict[str, object]] = []

        for current_day in trading_dates:
            latest_prices = pivot.loc[current_day].dropna()
            benchmark_price = float(latest_prices[benchmark_symbol])
            strategy_contribution = 0.0
            benchmark_contribution = 0.0

            if current_day >= first_strategy_day and total_contributed_strategy <= 0:
                cash += float(self.config.execution.initial_cash)
                total_contributed_strategy = float(self.config.execution.initial_cash)
                last_strategy_contribution_date = first_strategy_day.date()
                initial_targets = self.build_initial_seed_targets(
                    normalized_disclosures,
                    normalized_prices,
                    as_of_date=current_day,
                )
                initial_target_weights = initial_targets.set_index("symbol")["weight"].to_dict() if not initial_targets.empty else {}
                if initial_target_weights:
                    cash, positions, new_trades = self._simulate_rebalance(
                        current_day=current_day,
                        latest_prices=latest_prices,
                        positions=positions,
                        cash=cash,
                        target_weights=initial_target_weights,
                    )
                    trades.extend(new_trades)

            if current_day >= benchmark_first_day and benchmark_units <= 0:
                benchmark_units = float(self.config.execution.initial_cash / first_benchmark_price)
                total_contributed_benchmark = float(self.config.execution.initial_cash)
                last_benchmark_contribution_date = benchmark_first_day.date()

            signal_frame = self.build_signal_frame(
                normalized_disclosures,
                normalized_prices,
                as_of_date=current_day,
            )
            targets = (
                self.build_targets(signal_frame, positions, cash, as_of_date=current_day)
                if total_contributed_strategy > 0
                else pd.DataFrame(columns=["date", "symbol", "weight", "score", "reason"])
            )
            contribution_due = total_contributed_strategy > 0 and _is_contribution_due(
                current_day.date(),
                last_strategy_contribution_date,
                self.config.execution.contribution_day,
            )
            if contribution_due:
                strategy_contribution = float(self.config.execution.monthly_contribution)
                cash += strategy_contribution
                total_contributed_strategy += strategy_contribution
                last_strategy_contribution_date = current_day.date()
                if benchmark_units > 0 and _is_contribution_due(
                    current_day.date(),
                    last_benchmark_contribution_date,
                    self.config.execution.contribution_day,
                ):
                    benchmark_contribution = float(self.config.execution.monthly_contribution)
                    benchmark_units += benchmark_contribution / benchmark_price
                    total_contributed_benchmark += benchmark_contribution
                    last_benchmark_contribution_date = current_day.date()
            target_weights = targets.set_index("symbol")["weight"].to_dict() if not targets.empty else {}
            held_symbols = {symbol for symbol, state in positions.items() if float(state.get("quantity", 0.0)) > 1e-8}
            target_symbols = set(target_weights)

            # Daily evaluation does not imply daily trading. Trades only happen
            # when a contribution arrives or the active target set changes.
            needs_rebalance = bool(strategy_contribution > 0 and target_symbols)
            needs_rebalance = needs_rebalance or held_symbols != target_symbols

            if total_contributed_strategy > 0 and needs_rebalance:
                cash, positions, new_trades = self._simulate_rebalance(
                    current_day=current_day,
                    latest_prices=latest_prices,
                    positions=positions,
                    cash=cash,
                    target_weights=target_weights,
                )
                trades.extend(new_trades)

            market_value = sum(
                float(state["quantity"]) * float(latest_prices[symbol])
                for symbol, state in positions.items()
                if symbol in latest_prices.index
            )
            portfolio_value = cash + market_value
            benchmark_value = benchmark_units * benchmark_price if benchmark_units > 0 else 0.0
            curve_rows.append(
                {
                    "date": current_day,
                    "cash": cash,
                    "market_value": market_value,
                    "portfolio_value": portfolio_value,
                    "benchmark_value": benchmark_value,
                    "strategy_contribution": strategy_contribution,
                    "benchmark_contribution": benchmark_contribution,
                    "total_contributed_strategy": total_contributed_strategy,
                    "total_contributed_benchmark": total_contributed_benchmark,
                    "trade_count": len(trades),
                }
            )

        curve = pd.DataFrame(curve_rows)
        metrics = self._compute_metrics(curve, trade_count=len(trades))
        return SimulationState(
            cash=float(cash),
            positions=positions,
            benchmark_units=float(benchmark_units),
            total_contributed_strategy=float(total_contributed_strategy),
            total_contributed_benchmark=float(total_contributed_benchmark),
            last_strategy_contribution_date=last_strategy_contribution_date,
            last_benchmark_contribution_date=last_benchmark_contribution_date,
            trades=trades,
            curve=curve,
            metrics=metrics,
        )

    def _simulate_rebalance(
        self,
        current_day: pd.Timestamp,
        latest_prices: pd.Series,
        positions: dict[str, dict[str, float]],
        cash: float,
        target_weights: dict[str, float],
    ) -> tuple[float, dict[str, dict[str, float]], list[BacktestTrade]]:
        # The backtest uses the same sizing idea as the live bot: sell removed
        # names first, then buy toward the slot targets with cash clipping.
        trades: list[BacktestTrade] = []
        next_positions = {symbol: {"quantity": float(state["quantity"]), "avg_cost": float(state["avg_cost"])} for symbol, state in positions.items()}
        portfolio_value = cash + sum(
            float(state["quantity"]) * float(latest_prices[symbol])
            for symbol, state in next_positions.items()
            if symbol in latest_prices.index
        )

        current_weights = {symbol: target_weights.get(symbol, 0.0) for symbol in set(target_weights) | set(next_positions)}
        target_values = {symbol: portfolio_value * weight for symbol, weight in current_weights.items()}

        # Sells free cash before buys are attempted.
        sell_symbols = sorted(symbol for symbol in next_positions if symbol not in target_weights)
        for symbol in sell_symbols:
            if symbol not in latest_prices.index:
                continue
            price = float(latest_prices[symbol])
            current_qty = float(next_positions[symbol]["quantity"])
            gross_notional = current_qty * price
            if gross_notional < self.config.execution.min_trade_dollars:
                next_positions.pop(symbol, None)
                continue
            exec_price = price * (1 - self.config.execution.slippage_bps / 10_000.0)
            fees = gross_notional * (self.config.execution.commission_bps / 10_000.0)
            cash += current_qty * exec_price - fees
            next_positions.pop(symbol, None)
            trades.append(
                BacktestTrade(
                    date=current_day,
                    symbol=symbol,
                    side="sell",
                    gross_notional=gross_notional,
                )
            )

        buy_symbols = sorted(target_weights)
        for symbol in buy_symbols:
            if symbol not in latest_prices.index:
                continue
            price = float(latest_prices[symbol])
            current_qty = float(next_positions.get(symbol, {}).get("quantity", 0.0))
            target_value = float(target_values.get(symbol, 0.0))
            target_qty = target_value / price if price > 0 else 0.0
            delta_qty = target_qty - current_qty
            gross_notional = abs(delta_qty) * price
            if gross_notional < self.config.execution.min_trade_dollars or delta_qty <= 0:
                continue
            exec_price = price * (1 + self.config.execution.slippage_bps / 10_000.0)
            max_affordable_qty = cash / (exec_price * (1 + self.config.execution.commission_bps / 10_000.0)) if exec_price > 0 else 0.0
            trade_qty = min(delta_qty, max_affordable_qty)
            gross_notional = trade_qty * price
            if trade_qty <= 1e-8 or gross_notional < self.config.execution.min_trade_dollars:
                continue
            fees = gross_notional * (self.config.execution.commission_bps / 10_000.0)
            cash -= trade_qty * exec_price + fees
            new_qty = current_qty + trade_qty
            current_avg = float(next_positions.get(symbol, {}).get("avg_cost", 0.0))
            new_avg = ((current_qty * current_avg) + (trade_qty * exec_price)) / new_qty if new_qty > 0 else 0.0
            next_positions[symbol] = {"quantity": new_qty, "avg_cost": new_avg}
            trades.append(
                BacktestTrade(
                    date=current_day,
                    symbol=symbol,
                    side="buy",
                    gross_notional=gross_notional,
                )
            )

        return cash, next_positions, trades

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        # Stable zero payload used when the simulator cannot produce a curve.
        return {
            "ending_value": 0.0,
            "benchmark_ending_value": 0.0,
            "total_contributions_strategy": 0.0,
            "total_contributions_benchmark": 0.0,
            "net_profit": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "relative_performance_vs_benchmark": 0.0,
            "trade_count": 0.0,
        }

    def _compute_metrics(self, curve: pd.DataFrame, trade_count: int) -> dict[str, float]:
        # Metrics focus on the contribution-aware quantities the user actually
        # needs to audit this simple paper bot.
        if curve.empty:
            return self._empty_metrics()

        ending_value = float(curve["portfolio_value"].iloc[-1])
        benchmark_ending_value = float(curve["benchmark_value"].iloc[-1])
        total_contributions_strategy = float(curve["total_contributed_strategy"].iloc[-1])
        total_contributions_benchmark = float(curve["total_contributed_benchmark"].iloc[-1])
        net_profit = ending_value - total_contributions_strategy

        days = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days, 1)
        years = days / 365.25
        cagr = 0.0
        if years > 0 and total_contributions_strategy > 0:
            cagr = (ending_value / total_contributions_strategy) ** (1 / years) - 1

        rolling_max = curve["portfolio_value"].cummax()
        max_drawdown = float(((curve["portfolio_value"] / rolling_max) - 1.0).min())
        relative_performance = 0.0
        if benchmark_ending_value > 0:
            relative_performance = (ending_value / benchmark_ending_value) - 1.0

        return {
            "ending_value": ending_value,
            "benchmark_ending_value": benchmark_ending_value,
            "total_contributions_strategy": total_contributions_strategy,
            "total_contributions_benchmark": total_contributions_benchmark,
            "net_profit": net_profit,
            "cagr": float(cagr),
            "max_drawdown": max_drawdown,
            "relative_performance_vs_benchmark": float(relative_performance),
            "trade_count": float(trade_count),
        }


def format_metrics(metrics: dict[str, float]) -> str:
    # Compact CLI formatter used by backtest and paper-run output.
    ordered_keys = [
        "ending_value",
        "benchmark_ending_value",
        "total_contributions_strategy",
        "total_contributions_benchmark",
        "net_profit",
        "cagr",
        "max_drawdown",
        "relative_performance_vs_benchmark",
        "trade_count",
    ]
    lines: list[str] = []
    for key in ordered_keys:
        if key not in metrics:
            continue
        lines.append(f"{key}: {metrics[key]:.4f}")
    return "\n".join(lines)
