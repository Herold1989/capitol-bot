from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(slots=True)
class TradeDisclosure:
    # Canonical typed representation of one scraped disclosure.
    politician: str
    ticker: str
    asset_name: str
    traded_at: date
    published_at: date
    transaction_type: str
    size_low: float
    size_high: float
    filing_delay_days: int
    party: Optional[str] = None
    chamber: Optional[str] = None
    state: Optional[str] = None

    @property
    def notional_mid(self) -> float:
        # Use the midpoint because Capitol Trades provides value ranges, not exact sizes.
        return (self.size_low + self.size_high) / 2.0

    @property
    def signed_notional(self) -> float:
        # Positive for buys, negative for sells.
        sign = -1.0 if self.transaction_type.lower().startswith("sell") else 1.0
        return sign * self.notional_mid


@dataclass(slots=True)
class PositionTarget:
    # Desired post-rebalance holding emitted by the strategy.
    symbol: str
    weight: float
    score: float
    sector: Optional[str] = None
