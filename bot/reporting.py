from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import psycopg
from PIL import Image, ImageDraw, ImageFont


@dataclass(slots=True)
class DailyReport:
    # Denormalized payload used to render the short daily text summary.
    run_id: int
    run_date: str
    contribution_applied: bool
    strategy_contribution_amount: float
    benchmark_contribution_amount: float
    total_contributed_strategy: float
    total_contributed_benchmark: float
    portfolio_value: float
    benchmark_value: float | None
    benchmark_symbol: str
    cash: float
    market_value: float
    rebalance_executed: bool
    action_summary: str
    action_reason: str
    routine_status: dict[str, object]
    top_holdings: pd.DataFrame
    recent_trades: pd.DataFrame
    skipped_symbols: pd.DataFrame


def load_latest_daily_report(conn: psycopg.Connection) -> DailyReport | None:
    # Pull the latest persisted paper-run row and a few small supporting child tables.
    run = conn.execute(
        """
        SELECT id, run_date, contribution_applied, strategy_contribution_amount, benchmark_contribution_amount,
               total_contributed_strategy, total_contributed_benchmark,
               portfolio_value, benchmark_value, benchmark_symbol, cash, market_value,
               rebalance_executed, action_summary, action_reason, metrics
        FROM paper_runs
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if run is None:
        return None

    run_id = int(run["id"])
    top_holdings = pd.DataFrame(
        conn.execute(
            """
            SELECT symbol, asset_name, quantity, price, market_value, target_weight
            FROM position_snapshots
            WHERE run_id = %s
            ORDER BY market_value DESC, symbol
            LIMIT 8
            """,
            (run_id,),
        ).fetchall()
    )
    recent_trades = pd.DataFrame(
        conn.execute(
            """
            SELECT side, symbol, quantity, price, gross_notional, reason
            FROM paper_trades
            WHERE run_id = %s
            ORDER BY id
            LIMIT 8
            """,
            (run_id,),
        ).fetchall()
    )
    skipped_symbols = pd.DataFrame(
        conn.execute(
            """
            SELECT raw_symbol, reason
            FROM symbol_resolutions
            WHERE run_id = %s AND status <> 'ok'
            ORDER BY raw_symbol
            LIMIT 8
            """,
            (run_id,),
        ).fetchall()
    )

    return DailyReport(
        run_id=run_id,
        run_date=str(run["run_date"]),
        contribution_applied=bool(run["contribution_applied"]),
        strategy_contribution_amount=float(run["strategy_contribution_amount"]),
        benchmark_contribution_amount=float(run["benchmark_contribution_amount"]),
        total_contributed_strategy=float(run["total_contributed_strategy"]),
        total_contributed_benchmark=float(run["total_contributed_benchmark"]),
        portfolio_value=float(run["portfolio_value"]),
        benchmark_value=float(run["benchmark_value"]) if run["benchmark_value"] is not None else None,
        benchmark_symbol=str(run["benchmark_symbol"]),
        cash=float(run["cash"]),
        market_value=float(run["market_value"]),
        rebalance_executed=bool(run["rebalance_executed"]),
        action_summary=str(run["action_summary"] or ""),
        action_reason=str(run["action_reason"] or ""),
        routine_status=dict((run["metrics"] or {}).get("routine", {})),
        top_holdings=top_holdings,
        recent_trades=recent_trades,
        skipped_symbols=skipped_symbols,
    )


def load_portfolio_history(conn: psycopg.Connection) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT run_date, portfolio_value, benchmark_value, total_contributed_strategy,
               total_contributed_benchmark, cash, market_value
        FROM paper_runs
        WHERE command = 'paper-run'
        ORDER BY run_date, id
        """
    ).fetchall()
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_date",
                "portfolio_value",
                "benchmark_value",
                "total_contributed_strategy",
                "total_contributed_benchmark",
                "cash",
                "market_value",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["run_date"] = pd.to_datetime(frame["run_date"]).dt.date
    return frame


def render_portfolio_chart_svg(history: pd.DataFrame, width: int = 960, height: int = 520) -> str:
    if history.empty:
        return """<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520"><text x="40" y="60" font-family="Arial" font-size="18">No portfolio history yet</text></svg>"""

    frame = history.copy()
    frame["run_date"] = pd.to_datetime(frame["run_date"])
    for column in ["portfolio_value", "benchmark_value", "total_contributed_strategy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["run_date", "portfolio_value"])
    if frame.empty:
        return """<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520"><text x="40" y="60" font-family="Arial" font-size="18">No portfolio history yet</text></svg>"""

    margin_left, margin_right, margin_top, margin_bottom = 76, 34, 42, 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    date_min = frame["run_date"].min()
    date_max = frame["run_date"].max()
    date_span = max((date_max - date_min).days, 1)
    value_columns = ["portfolio_value", "benchmark_value", "total_contributed_strategy"]
    y_min = float(frame[value_columns].min(skipna=True).min())
    y_max = float(frame[value_columns].max(skipna=True).max())
    y_pad = max((y_max - y_min) * 0.08, 50.0)
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad
    y_span = max(y_max - y_min, 1.0)

    def x_pos(ts: pd.Timestamp) -> float:
        return margin_left + ((ts - date_min).days / date_span) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - ((value - y_min) / y_span)) * plot_height

    def polyline(column: str) -> str:
        points = []
        for _, row in frame.dropna(subset=[column]).iterrows():
            points.append(f"{x_pos(row['run_date']):.1f},{y_pos(float(row[column])):.1f}")
        return " ".join(points)

    y_ticks = [y_min + (y_span * idx / 4) for idx in range(5)]
    grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y_pos(value):.1f}" x2="{width - margin_right}" y2="{y_pos(value):.1f}" stroke="#e5e7eb" />'
        f'<text x="{margin_left - 10}" y="{y_pos(value) + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">${value:,.0f}</text>'
        for value in y_ticks
    )
    first_date = date_min.strftime("%Y-%m-%d")
    last_date = date_max.strftime("%Y-%m-%d")
    latest = frame.iloc[-1]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin_left}" y="26" font-family="Arial" font-size="20" font-weight="700" fill="#111827">Portfolio development</text>
  <text x="{width - margin_right}" y="26" text-anchor="end" font-family="Arial" font-size="13" fill="#4b5563">Latest ${float(latest['portfolio_value']):,.2f}</text>
  {grid}
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af" />
  <polyline points="{polyline('total_contributed_strategy')}" fill="none" stroke="#6b7280" stroke-width="2" stroke-dasharray="6 5" />
  <polyline points="{polyline('benchmark_value')}" fill="none" stroke="#2563eb" stroke-width="3" />
  <polyline points="{polyline('portfolio_value')}" fill="none" stroke="#059669" stroke-width="3" />
  <text x="{margin_left}" y="{height - 42}" font-family="Arial" font-size="12" fill="#4b5563">{first_date}</text>
  <text x="{width - margin_right}" y="{height - 42}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{last_date}</text>
  <rect x="{margin_left}" y="{height - 28}" width="14" height="4" fill="#059669" /><text x="{margin_left + 20}" y="{height - 23}" font-family="Arial" font-size="12" fill="#111827">Capitol portfolio</text>
  <rect x="{margin_left + 160}" y="{height - 28}" width="14" height="4" fill="#2563eb" /><text x="{margin_left + 180}" y="{height - 23}" font-family="Arial" font-size="12" fill="#111827">Benchmark</text>
  <rect x="{margin_left + 285}" y="{height - 28}" width="14" height="4" fill="#6b7280" /><text x="{margin_left + 305}" y="{height - 23}" font-family="Arial" font-size="12" fill="#111827">Contributed cash</text>
</svg>
"""


def write_portfolio_chart_png(history: pd.DataFrame, path: str, width: int = 960, height: int = 520) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bold_font = ImageFont.load_default()

    if history.empty:
        draw.text((40, 60), "No portfolio history yet", fill="#111827", font=font)
        image.save(path)
        return

    frame = history.copy()
    frame["run_date"] = pd.to_datetime(frame["run_date"])
    for column in ["portfolio_value", "benchmark_value", "total_contributed_strategy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["run_date", "portfolio_value"])
    if frame.empty:
        draw.text((40, 60), "No portfolio history yet", fill="#111827", font=font)
        image.save(path)
        return

    margin_left, margin_right, margin_top, margin_bottom = 76, 34, 42, 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    date_min = frame["run_date"].min()
    date_max = frame["run_date"].max()
    date_span = max((date_max - date_min).days, 1)
    value_columns = ["portfolio_value", "benchmark_value", "total_contributed_strategy"]
    y_min = float(frame[value_columns].min(skipna=True).min())
    y_max = float(frame[value_columns].max(skipna=True).max())
    y_pad = max((y_max - y_min) * 0.08, 50.0)
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad
    y_span = max(y_max - y_min, 1.0)

    def x_pos(ts: pd.Timestamp) -> float:
        return margin_left + ((ts - date_min).days / date_span) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - ((value - y_min) / y_span)) * plot_height

    draw.text((margin_left, 18), "Portfolio development", fill="#111827", font=bold_font)
    latest = frame.iloc[-1]
    draw.text((width - margin_right - 150, 18), f"Latest ${float(latest['portfolio_value']):,.2f}", fill="#4b5563", font=font)

    for idx in range(5):
        value = y_min + (y_span * idx / 4)
        y = y_pos(value)
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        draw.text((8, y - 6), f"${value:,.0f}", fill="#4b5563", font=font)

    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#9ca3af", width=1)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="#9ca3af", width=1)

    def draw_series(column: str, color: str, dashed: bool = False) -> None:
        points = [
            (x_pos(row["run_date"]), y_pos(float(row[column])))
            for _, row in frame.dropna(subset=[column]).iterrows()
        ]
        if len(points) < 2:
            return
        if not dashed:
            draw.line(points, fill=color, width=3)
            return
        for start, end in zip(points, points[1:]):
            draw.line((start, end), fill=color, width=2)

    draw_series("total_contributed_strategy", "#6b7280", dashed=True)
    draw_series("benchmark_value", "#2563eb")
    draw_series("portfolio_value", "#059669")

    draw.text((margin_left, height - 45), date_min.strftime("%Y-%m-%d"), fill="#4b5563", font=font)
    draw.text((width - margin_right - 72, height - 45), date_max.strftime("%Y-%m-%d"), fill="#4b5563", font=font)
    legend_y = height - 28
    draw.rectangle((margin_left, legend_y, margin_left + 14, legend_y + 4), fill="#059669")
    draw.text((margin_left + 20, legend_y - 6), "Capitol portfolio", fill="#111827", font=font)
    draw.rectangle((margin_left + 160, legend_y, margin_left + 174, legend_y + 4), fill="#2563eb")
    draw.text((margin_left + 180, legend_y - 6), "Benchmark", fill="#111827", font=font)
    draw.rectangle((margin_left + 285, legend_y, margin_left + 299, legend_y + 4), fill="#6b7280")
    draw.text((margin_left + 305, legend_y - 6), "Contributed cash", fill="#111827", font=font)
    image.save(path)


def _format_money(value: float | None) -> str:
    # Shared currency formatter for all message lines.
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _format_pct(value: float | None) -> str:
    # Shared percentage formatter for the relative-performance line.
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_daily_message(report: DailyReport) -> str:
    # The report intentionally reads like a compact status text rather than a full blotter.
    relative_performance = None
    if report.benchmark_value not in (None, 0):
        relative_performance = (report.portfolio_value / report.benchmark_value) - 1.0

    if report.contribution_applied:
        contribution_line = (
            f"Contribution applied: yes ({_format_money(report.strategy_contribution_amount)} strategy, "
            f"{_format_money(report.benchmark_contribution_amount)} benchmark)"
        )
    else:
        contribution_line = "Contribution applied: no"

    if not report.top_holdings.empty:
        holdings = ", ".join(
            (
                f"{row['symbol']} ({row['asset_name']}) {_format_money(float(row['market_value']))}"
                if row.get("asset_name") and str(row.get("asset_name")).strip() and str(row.get("asset_name")) != str(row["symbol"])
                else f"{row['symbol']} {_format_money(float(row['market_value']))}"
            )
            for _, row in report.top_holdings.head(3).iterrows()
        )
        holdings = f"{holdings}, cash {_format_money(report.cash)}"
    else:
        holdings = f"cash {_format_money(report.cash)}"

    trades_line = report.action_summary if report.action_summary else ("none" if report.recent_trades.empty else "see ledger")
    stock_value = max(report.market_value, 0.0)
    lines = [
        f"Capitol bot update {report.run_date}",
        contribution_line,
        f"Benchmark ETF {_format_money(report.benchmark_value)} on {_format_money(report.total_contributed_benchmark)} contributed",
        f"Capitol stocks {_format_money(stock_value)} on {_format_money(report.total_contributed_strategy)} contributed",
        f"Capitol total {_format_money(report.portfolio_value)}",
        f"Relative performance Capitol vs benchmark: {_format_pct(relative_performance)}",
        f"Cash balance: {_format_money(report.cash)}",
        f"Trades today: {trades_line}",
        f"Reason: {report.action_reason or 'no explicit reason recorded'}",
        f"Holdings: {holdings}",
    ]

    if report.routine_status:
        lines.append(f"Data status: {report.routine_status.get('reason', 'fresh')}")

    if not report.skipped_symbols.empty:
        skipped = ", ".join(f"{row['raw_symbol']} ({row['reason']})" for _, row in report.skipped_symbols.head(3).iterrows())
        lines.append(f"Skipped/unresolved: {skipped}")

    lines.append("Not investment advice.")
    return "\n".join(lines)
