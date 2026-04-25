from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf

from bot.symbol_resolver import SymbolResolution, SymbolResolver


class MarketDataClient:
    def __init__(self, provider_priority: Iterable[str] | None = None, min_history_rows: int = 80) -> None:
        # Provider order matters because resolution stops at the first usable match.
        self.provider_priority = [provider.lower() for provider in (provider_priority or ["yfinance"])]
        self.min_history_rows = min_history_rows
        self.symbol_resolver = SymbolResolver()

    @staticmethod
    def _trim_history(frame: pd.DataFrame, start: date | str, end: date | str | None = None) -> pd.DataFrame:
        # Normalize dates and clip the history to the configured window.
        if frame.empty:
            return frame
        trimmed = frame.copy()
        trimmed["date"] = pd.to_datetime(trimmed["date"]).dt.normalize()
        start_ts = pd.Timestamp(start)
        if end is None:
            return trimmed[trimmed["date"] >= start_ts]
        end_ts = pd.Timestamp(end)
        return trimmed[(trimmed["date"] >= start_ts) & (trimmed["date"] <= end_ts)]

    def _is_valid_history(self, frame: pd.DataFrame) -> bool:
        # Require enough rows to support momentum and volatility features.
        if frame.empty:
            return False
        valid_rows = frame["close"].notna().sum()
        return int(valid_rows) >= self.min_history_rows

    def _fetch_symbol_yfinance(self, symbol: str, start: date | str, end: date | str | None = None) -> pd.DataFrame:
        # Use auto-adjusted closes so returns naturally account for splits/dividends.
        yf_frame = yf.download(
            symbol,
            start=str(start),
            end=str(end) if end else None,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if yf_frame.empty:
            return pd.DataFrame(columns=["date", "close", "volume"])
        if isinstance(yf_frame.columns, pd.MultiIndex):
            # Flatten occasional MultiIndex outputs from yfinance.
            yf_frame.columns = [col[0] if isinstance(col, tuple) else col for col in yf_frame.columns]
        frame = yf_frame.reset_index().rename(columns={"Date": "date", "Close": "close"})
        if "close" not in frame.columns:
            return pd.DataFrame(columns=["date", "close", "volume"])
        if "Volume" in frame.columns:
            frame = frame.rename(columns={"Volume": "volume"})
        if "volume" not in frame.columns:
            frame["volume"] = pd.NA
        frame = frame[["date", "close", "volume"]].copy()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        return frame.dropna(subset=["close"])

    async def _fetch_from_provider(
        self,
        provider: str,
        candidate: str,
        start: date | str,
        end: date | str | None,
    ) -> pd.DataFrame:
        if provider == "yfinance":
            # yfinance is blocking, so run it off the event loop.
            return await asyncio.to_thread(self._fetch_symbol_yfinance, candidate, start, end)
        raise ValueError(f"Unsupported provider: {provider}")

    async def resolve_and_fetch_history(
        self,
        symbols: Iterable[str],
        start: date | str,
        end: date | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Deduplicate inputs before resolution attempts.
        symbols = sorted(set(symbol for symbol in symbols if symbol))
        frames: list[pd.DataFrame] = []
        resolutions: list[SymbolResolution] = []
        for raw_symbol in symbols:
            skip_reason = self.symbol_resolver.skip_reason(raw_symbol)
            if skip_reason:
                # Explicitly record skipped names for audit and reporting.
                resolutions.append(
                    SymbolResolution(
                        raw_symbol=raw_symbol,
                        resolved_symbol=None,
                        provider=None,
                        status="skipped",
                        reason=skip_reason,
                    )
                )
                continue

            candidates = self.symbol_resolver.candidate_symbols(raw_symbol)
            last_reason = "no_provider_match"
            matched = False
            for provider in self.provider_priority:
                for candidate in candidates:
                    try:
                        # Try candidates in order until one returns enough history.
                        frame = await self._fetch_from_provider(provider, candidate, start, end)
                    except Exception as exc:
                        last_reason = f"{provider}_error:{type(exc).__name__}"
                        continue
                    frame = self._trim_history(frame, start, end)
                    if not self._is_valid_history(frame):
                        last_reason = f"{provider}_insufficient_history"
                        continue
                    frame = frame.assign(symbol=raw_symbol, resolved_symbol=candidate, provider=provider)
                    frames.append(frame)
                    resolutions.append(
                        SymbolResolution(
                            raw_symbol=raw_symbol,
                            resolved_symbol=candidate,
                            provider=provider,
                            status="ok",
                            reason="resolved",
                            history_rows=len(frame),
                        )
                    )
                    matched = True
                    break
                if matched:
                    break
            if not matched:
                # Keep the last failure reason to help diagnose resolver gaps.
                resolutions.append(
                    SymbolResolution(
                        raw_symbol=raw_symbol,
                        resolved_symbol=None,
                        provider=None,
                        status="unresolved",
                        reason=last_reason,
                    )
                )
        if not frames:
            empty_prices = pd.DataFrame(columns=["date", "close", "volume", "symbol", "resolved_symbol", "provider"])
            resolution_frame = pd.DataFrame([asdict(resolution) for resolution in resolutions])
            return empty_prices, resolution_frame
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
        resolution_frame = pd.DataFrame([asdict(resolution) for resolution in resolutions])
        return combined.sort_values(["symbol", "date"]).reset_index(drop=True), resolution_frame

    async def fetch_history(self, symbols: Iterable[str], start: date | str, end: date | str | None = None) -> pd.DataFrame:
        # Convenience wrapper for callers that do not need resolution metadata.
        prices, _ = await self.resolve_and_fetch_history(symbols, start, end)
        return prices
