from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    # Historical price/disclosure window used by both the live bot and the
    # walk-forward backtest.
    start_date: str
    end_date: str | None = None
    # The strategy and benchmark both begin taking capital risk on or after
    # this date, while start_date can remain earlier to supply indicator history.
    portfolio_start_date: str = "2026-01-01"
    # The benchmark starts as the user-facing ISIN and is resolved later into a
    # provider symbol through SymbolResolver.
    benchmark_symbol: str = "IE00B5BMR087"
    benchmark_report_symbols: list[str] = Field(default_factory=lambda: ["SPY", "VOO"])
    # Benchmark accounting can begin later than the disclosure history window if
    # the user wants the contribution-matched comparison to start at a specific date.
    benchmark_start_date: str = "2026-01-01"
    # Published Capitol disclosures are only considered for this many calendar days.
    signal_lookback_days: int = 45
    # The trend filter requires the latest close to sit above this moving average.
    moving_average_days: int = 50
    # The strategy is intentionally small and conservative.
    max_positions: int = 3
    min_buy_disclosures: int = 2
    # The live/backtest seed starts from the most-bought Capitol names from the
    # prior calendar year rather than from cash.
    initial_selection_year: int = 2025
    initial_top_n: int = 5
    # The bot evaluates every daily run but should only trade when the signal
    # set or available contribution cash makes action necessary.
    rebalance_frequency: str = "D"
    # Prefer deploying capital across the active names instead of leaving empty
    # slots in cash, unless no names qualify or existing holdings hit sell rules.
    allow_cash_buffer: bool = False
    # Existing holdings may be carried only while they remain technically valid
    # and their last disclosure is not too old.
    max_stale_hold_days: int = 90
    # Per-name cap keeps the paper book from becoming a single-stock bet when
    # fewer than max_positions qualify.
    max_single_position_weight: float = 0.50
    # Optional liquidity guard; only active when provider data includes volume.
    min_avg_dollar_volume: float = 0.0
    min_rebalance_drift: float = 0.05
    min_replacement_score_advantage: float = 0.75
    disclosure_delay_half_life_days: float = 30.0
    stop_loss_pct: float = 0.20
    reliability_lookahead_days: int = 20


class DataConfig(BaseModel):
    # Provider priority is ordered because the resolver stops at the first
    # source that returns a usable history.
    provider_priority: list[str] = Field(default_factory=lambda: ["yfinance"])
    min_history_rows: int = 80
    issuer_pages: int = 3
    trade_pages: int = 10
    # The one-time seed basket may need much deeper pagination to rank the
    # prior year's most-bought Capitol names from the public site.
    seed_trade_pages: int = 1600
    request_timeout_seconds: int = 30
    user_agent: str = "capitol-paper-bot/0.2"
    cache_dir: str = "data/cache"
    recent_disclosures_cache_hours: int = 12
    price_cache_hours: int = 18
    max_price_staleness_days: int = 5
    max_disclosure_staleness_days: int = 14
    use_market_calendar_staleness: bool = True


class ExecutionConfig(BaseModel):
    # Live paper state begins with a small starter account instead of the old
    # allocator's six-figure notional.
    initial_cash: float = 5_000.0
    # Fresh cash is added on the first eligible run each new month.
    monthly_contribution: float = 250.0
    contribution_day: int = 1
    slippage_bps: float = 10.0
    commission_bps: float = 2.0
    # Tiny trades are skipped to keep the ledger clean and avoid noise.
    min_trade_dollars: float = 25.0


class AppConfig(BaseModel):
    strategy: StrategyConfig
    data: DataConfig
    execution: ExecutionConfig


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if not config_path.exists():
        fallback_path = Path("/app/default_config") / config_path.name
        if fallback_path.exists():
            return fallback_path
    return config_path


def load_config(path: str | Path) -> AppConfig:
    # Parse YAML once and validate every nested section before the run starts.
    config_path = resolve_config_path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    return AppConfig.model_validate(raw)
