from __future__ import annotations

from dataclasses import dataclass
import re


MUTUAL_FUND_LIKE_RE = re.compile(r"^[A-Z]{4}X$")
INVALID_CHAR_RE = re.compile(r"[^A-Z0-9./\-]")

MANUAL_ALIASES = {
    # Explicit fixes for known Capitol Trades vs provider symbol mismatches.
    "IE00B5BMR087": "CSPX.L",
    "BRK/B": "BRK-B",
    "BRK.B": "BRK-B",
    "BF/B": "BF-B",
    "BF.B": "BF-B",
}

NON_EQUITY_SYMBOLS = {
    # Symbols that represent indexes or derivatives instead of spot equities/ETFs.
    "DJX": "index_or_derivative",
    "NDX": "index_or_derivative",
    "RUT": "index_or_derivative",
    "SPX": "index_or_derivative",
    "VIX": "index_or_derivative",
    "XSP": "index_or_derivative",
}


@dataclass(slots=True)
class SymbolResolution:
    # Audit record describing how one raw symbol was handled.
    raw_symbol: str
    resolved_symbol: str | None
    provider: str | None
    status: str
    reason: str
    history_rows: int = 0


class SymbolResolver:
    def normalize(self, symbol: str) -> str:
        # Strip common exchange suffixes and normalize case for matching.
        normalized = symbol.upper().strip()
        normalized = normalized.replace(":US", "")
        return normalized

    def skip_reason(self, symbol: str) -> str | None:
        # Reject obviously invalid or out-of-universe symbols before data lookup.
        normalized = self.normalize(symbol)
        if not normalized:
            return "empty_symbol"
        if normalized in NON_EQUITY_SYMBOLS:
            return NON_EQUITY_SYMBOLS[normalized]
        if INVALID_CHAR_RE.search(normalized):
            return "invalid_characters"
        if MUTUAL_FUND_LIKE_RE.match(normalized):
            return "mutual_fund_like"
        return None

    def candidate_symbols(self, symbol: str) -> list[str]:
        # Generate a short ordered list of plausible provider symbols.
        normalized = self.normalize(symbol)
        if not normalized:
            return []

        candidates: list[str] = []

        def add(candidate: str) -> None:
            # Preserve candidate priority while avoiding duplicates.
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add(MANUAL_ALIASES.get(normalized, normalized))
        add(normalized)
        add(normalized.replace("/", "-"))
        add(normalized.replace(".", "-"))
        add(normalized.replace(".", ""))
        add(normalized.replace("/", ""))
        return candidates
