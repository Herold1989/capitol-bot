from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from bot.config import AppConfig


@dataclass(slots=True)
class RoutineStatus:
    disclosure_count: int
    latest_disclosure_date: str | None
    attempted_symbol_count: int
    resolved_symbol_count: int
    unresolved_symbol_count: int
    latest_price_date: str | None
    benchmark_latest_price_date: str | None
    provider_error_count: int
    stale_price: bool
    stale_disclosures: bool
    degraded: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cache_is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    modified_at = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    age_hours = (pd.Timestamp.utcnow() - modified_at).total_seconds() / 3600.0
    return age_hours <= max_age_hours


def read_cached_frame(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates or [])


def write_cached_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _elapsed_market_days(start: pd.Timestamp, end: pd.Timestamp, use_market_calendar: bool) -> int:
    if end <= start:
        return 0
    if not use_market_calendar:
        return int((end - start).days)
    trading_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return max(len(pd.date_range(start=start + trading_day, end=end, freq=trading_day)), 0)


def build_routine_status(
    disclosures: pd.DataFrame,
    prices: pd.DataFrame,
    symbol_resolution: pd.DataFrame,
    benchmark_symbol: str,
    config: AppConfig,
    as_of_date: pd.Timestamp | None = None,
) -> RoutineStatus:
    now = pd.Timestamp(as_of_date or pd.Timestamp.utcnow()).tz_localize(None).normalize()
    disclosure_count = int(len(disclosures))
    latest_disclosure_date = None
    stale_disclosures = True
    if not disclosures.empty and "published_at" in disclosures.columns:
        latest_disclosure_ts = pd.to_datetime(disclosures["published_at"], errors="coerce").max()
        if pd.notna(latest_disclosure_ts):
            latest_disclosure_ts = latest_disclosure_ts.tz_localize(None).normalize()
            latest_disclosure_date = str(latest_disclosure_ts.date())
            stale_disclosures = _elapsed_market_days(
                latest_disclosure_ts,
                now,
                bool(config.data.use_market_calendar_staleness),
            ) > int(config.data.max_disclosure_staleness_days)

    latest_price_date = None
    benchmark_latest_price_date = None
    stale_price = True
    if not prices.empty and "date" in prices.columns:
        price_dates = pd.to_datetime(prices["date"], errors="coerce")
        latest_price_ts = price_dates.max()
        if pd.notna(latest_price_ts):
            latest_price_ts = latest_price_ts.tz_localize(None).normalize()
            latest_price_date = str(latest_price_ts.date())
            stale_price = _elapsed_market_days(
                latest_price_ts,
                now,
                bool(config.data.use_market_calendar_staleness),
            ) > int(config.data.max_price_staleness_days)
        benchmark_prices = prices.loc[prices["symbol"] == benchmark_symbol, "date"] if "symbol" in prices.columns else pd.Series(dtype=object)
        benchmark_latest_ts = pd.to_datetime(benchmark_prices, errors="coerce").max()
        if pd.notna(benchmark_latest_ts):
            benchmark_latest_ts = benchmark_latest_ts.tz_localize(None).normalize()
            benchmark_latest_price_date = str(benchmark_latest_ts.date())
            stale_price = stale_price or _elapsed_market_days(
                benchmark_latest_ts,
                now,
                bool(config.data.use_market_calendar_staleness),
            ) > int(config.data.max_price_staleness_days)

    attempted_symbol_count = int(len(symbol_resolution)) if not symbol_resolution.empty else 0
    resolved_symbol_count = (
        int((symbol_resolution["status"] == "ok").sum())
        if not symbol_resolution.empty and "status" in symbol_resolution.columns
        else 0
    )
    unresolved_symbol_count = max(attempted_symbol_count - resolved_symbol_count, 0)
    provider_error_count = (
        int(symbol_resolution["reason"].astype(str).str.contains("_error:", regex=False).sum())
        if not symbol_resolution.empty and "reason" in symbol_resolution.columns
        else 0
    )
    degraded = bool(stale_price or stale_disclosures or provider_error_count > 0 or unresolved_symbol_count > 0)
    reasons = []
    if stale_price:
        reasons.append("price data is stale")
    if stale_disclosures:
        reasons.append("disclosure feed is stale")
    if provider_error_count:
        reasons.append(f"{provider_error_count} provider errors")
    if unresolved_symbol_count:
        reasons.append(f"{unresolved_symbol_count} unresolved symbols")
    return RoutineStatus(
        disclosure_count=disclosure_count,
        latest_disclosure_date=latest_disclosure_date,
        attempted_symbol_count=attempted_symbol_count,
        resolved_symbol_count=resolved_symbol_count,
        unresolved_symbol_count=unresolved_symbol_count,
        latest_price_date=latest_price_date,
        benchmark_latest_price_date=benchmark_latest_price_date,
        provider_error_count=provider_error_count,
        stale_price=stale_price,
        stale_disclosures=stale_disclosures,
        degraded=degraded,
        reason=", ".join(reasons) if reasons else "fresh",
    )
