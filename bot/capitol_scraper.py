from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from bot.models import TradeDisclosure

DATE_RE = re.compile(r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})")
TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9\.\-]{0,9}):US\b")


def _parse_human_money(value: str) -> float:
    # Convert compact strings such as "1.2M" into raw numeric dollar values.
    value = value.replace(",", "").strip().upper()
    multiplier = 1.0
    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000.0
        value = value[:-1]
    return float(value) * multiplier


def _parse_money_range(text: str) -> tuple[float, float]:
    # Capitol Trades publishes trade sizes as low-high ranges.
    cleaned = text.replace(" ", "")
    parts = re.split(r"[–-]", cleaned)
    if len(parts) != 2:
        return 1_000.0, 15_000.0
    return _parse_human_money(parts[0]), _parse_human_money(parts[1])


def _parse_date(text: str) -> date:
    # Support both abbreviated and full month names seen on the site.
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {text}")


def _extract_json_array(payload: str, marker: str) -> list[dict[str, object]]:
    # The live trades page embeds the rows inside a Next.js streamed payload.
    # Extract the JSON array following a known marker by balancing brackets
    # rather than relying on a fragile regex that can be confused by nesting.
    for candidate in (payload, payload.replace('\\"', '"').replace("\\n", "\n")):
        marker_index = candidate.find(marker)
        if marker_index == -1:
            continue
        array_start = candidate.find("[", marker_index + len(marker))
        if array_start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(array_start, len(candidate)):
            char = candidate[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    try:
                        extracted = json.loads(candidate[array_start : idx + 1])
                    except json.JSONDecodeError:
                        break
                    return extracted if isinstance(extracted, list) else []
    return []


class CapitolTradesScraper:
    def __init__(self, user_agent: str, timeout_seconds: int = 30) -> None:
        # Reuse one session so headers and TCP connections stay consistent.
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout_seconds = timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch(self, url: str) -> str:
        # Retry transient fetch failures because the site can be inconsistent.
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def fetch_recent_trades(self, pages: int = 10) -> list[TradeDisclosure]:
        # Collect row-level disclosures from the recent-trades listing pages.
        disclosures: list[TradeDisclosure] = []
        for page in range(1, pages + 1):
            url = f"https://www.capitoltrades.com/trades?page={page}"
            html = self._fetch(url)
            disclosures.extend(self._parse_trade_listing(html))
        return disclosures

    def fetch_trades_since(self, min_published_date: date, max_pages: int = 10) -> list[TradeDisclosure]:
        # The first live seed and the walk-forward backtest need enough Capitol
        # history to cover the portfolio start plus the lookback window. Stop
        # paging once the scrape reaches that published-date threshold.
        disclosures: list[TradeDisclosure] = []
        for page in range(1, max_pages + 1):
            url = f"https://www.capitoltrades.com/trades?page={page}"
            html = self._fetch(url)
            page_disclosures = list(self._parse_trade_listing(html))
            if not page_disclosures:
                break
            disclosures.extend(page_disclosures)
            earliest_page_date = min(trade.published_at for trade in page_disclosures)
            if earliest_page_date <= min_published_date:
                break
        return [trade for trade in disclosures if trade.published_at >= min_published_date]

    def fetch_top_issuers_frame(self, pages: int = 3) -> pd.DataFrame:
        # Collect issuer-level popularity/volume context from the issuer pages.
        rows: list[dict[str, object]] = []
        for page in range(1, pages + 1):
            url = f"https://www.capitoltrades.com/issuers?page={page}"
            html = self._fetch(url)
            rows.extend(self._parse_issuer_listing(html))
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        # Convert issuer aggregates into disclosure-like rows so the strategy can
        # merge them with parsed trade disclosures using one schema.
        frame["politician"] = "aggregate_capitoltrades"
        frame["traded_at"] = today
        frame["published_at"] = today
        frame["transaction_type"] = "buy"
        frame["size_low"] = frame["volume"] * 0.5
        frame["size_high"] = frame["volume"] * 1.5
        frame["notional_mid"] = frame["volume"]
        frame["signed_notional"] = frame["volume"]
        frame["filing_delay_days"] = 0
        frame["party"] = None
        frame["chamber"] = None
        frame["state"] = None
        return frame

    def _parse_trade_listing(self, html: str) -> Iterable[TradeDisclosure]:
        # Prefer the structured Next.js payload because the current site renders
        # trade rows there. Keep the old text parser as a fallback in case the
        # page briefly regresses to the prior layout.
        structured = list(self._parse_trade_listing_structured(html))
        if structured:
            return structured
        # Parse the public page's text layout rather than relying on a stable API.
        soup = BeautifulSoup(html, "lxml")
        lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
        results: list[TradeDisclosure] = []
        meta_re = re.compile(r"^(Republican|Democrat|Other)\s+(Senate|House)\s+[A-Z]{2}$")
        i = 0
        while i < len(lines) - 13:
            politician = lines[i]
            meta = lines[i + 1]
            if not meta_re.match(meta):
                # Slide one line forward until a valid record boundary is found.
                i += 1
                continue
            asset_name = lines[i + 2]
            ticker_line = lines[i + 3]
            published_text = f"{lines[i + 4]} {lines[i + 5]}"
            traded_text = f"{lines[i + 6]} {lines[i + 7]}"
            if lines[i + 8].lower() != "days":
                i += 1
                continue
            filing_delay_raw = lines[i + 9]
            owner_line = lines[i + 10]
            tx_type = lines[i + 11].lower()
            size_text = lines[i + 12]
            if tx_type not in {"buy", "sell"}:
                i += 1
                continue
            ticker_match = TICKER_RE.search(ticker_line)
            ticker = ticker_match.group(1) if ticker_match else None
            party = "Republican" if "Republican" in meta else "Democrat" if "Democrat" in meta else None
            chamber = "Senate" if "Senate" in meta else "House" if "House" in meta else None
            state = meta[-2:] if len(meta) >= 2 else None
            try:
                published_at = _parse_date(published_text)
                traded_at = _parse_date(traded_text)
                filing_delay = int(filing_delay_raw)
            except (ValueError, TypeError):
                # Bad rows are skipped instead of killing the whole scrape.
                i += 1
                continue
            low, high = _parse_money_range(size_text)
            results.append(
                TradeDisclosure(
                    politician=politician,
                    ticker=ticker or "",
                    asset_name=asset_name,
                    traded_at=traded_at,
                    published_at=published_at,
                    transaction_type=tx_type,
                    size_low=low,
                    size_high=high,
                    filing_delay_days=filing_delay,
                    party=party,
                    chamber=chamber,
                    state=state,
                )
            )
            i += 14
        return results

    def _parse_trade_listing_structured(self, html: str) -> Iterable[TradeDisclosure]:
        # The live page contains the trade table inside a JSON `data` array
        # embedded in the streamed Next.js response. Reading that structure is
        # far more stable than scraping rendered text rows.
        rows = _extract_json_array(html, '"data":')
        results: list[TradeDisclosure] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tx_type = str(row.get("txType") or "").lower().strip()
            if tx_type not in {"buy", "sell"}:
                continue
            issuer = row.get("issuer") if isinstance(row.get("issuer"), dict) else {}
            politician = row.get("politician") if isinstance(row.get("politician"), dict) else {}
            ticker_blob = str(issuer.get("issuerTicker") or "").strip()
            ticker_match = TICKER_RE.search(ticker_blob)
            ticker = ticker_match.group(1) if ticker_match else ticker_blob.split(":")[0].strip()
            issuer_name = str(issuer.get("issuerName") or ticker or "").strip()
            first_name = str(politician.get("firstName") or "").strip()
            last_name = str(politician.get("lastName") or "").strip()
            nickname = str(politician.get("nickname") or "").strip()
            politician_name = " ".join(part for part in [first_name, nickname, last_name] if part).strip()
            pub_date_raw = str(row.get("pubDate") or "").strip()
            trade_date_raw = str(row.get("txDate") or "").strip()
            if not pub_date_raw or not trade_date_raw or not ticker:
                continue
            try:
                published_at = datetime.fromisoformat(pub_date_raw.replace("Z", "+00:00")).date()
                traded_at = datetime.fromisoformat(trade_date_raw).date()
                filing_delay_days = int(row.get("reportingGap") or 0)
                value = float(row.get("value") or 0.0)
            except (TypeError, ValueError):
                continue
            # The structured payload exposes a single midpoint-like dollar value
            # rather than the legacy low-high text bucket, so use it directly.
            results.append(
                TradeDisclosure(
                    politician=politician_name or "Unknown Politician",
                    ticker=ticker,
                    asset_name=issuer_name,
                    traded_at=traded_at,
                    published_at=published_at,
                    transaction_type=tx_type,
                    size_low=value,
                    size_high=value,
                    filing_delay_days=filing_delay_days,
                    party=str(politician.get("party") or "").title() or None,
                    chamber=str(row.get("chamber") or politician.get("chamber") or "").title() or None,
                    state=str(politician.get("_stateId") or "").upper() or None,
                )
            )
        return results

    def _parse_issuer_listing(self, html: str) -> list[dict[str, object]]:
        # Issuer pages have a different text layout but the same no-API parsing approach.
        soup = BeautifulSoup(html, "lxml")
        lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
        rows: list[dict[str, object]] = []
        i = 0
        while i < len(lines) - 9:
            asset_name = lines[i]
            ticker_match = TICKER_RE.search(lines[i + 1])
            if not ticker_match:
                i += 1
                continue
            ticker = ticker_match.group(1)
            try:
                last_traded = _parse_date(f"{lines[i + 2]} {lines[i + 3]}")
            except ValueError:
                i += 1
                continue
            volume_text = lines[i + 4]
            trades_text = lines[i + 5]
            politicians_text = lines[i + 6]
            sector = lines[i + 7]
            price_blob = lines[i + 9] if i + 9 < len(lines) else ""
            try:
                volume = _parse_human_money(volume_text)
                trades = int(trades_text.replace(",", ""))
                politicians = int(politicians_text.replace(",", ""))
            except ValueError:
                i += 1
                continue
            price_blob = next(
                (
                    candidate
                    for candidate in lines[i + 8 : i + 12]
                    if candidate != "Loading..." and candidate != "N/A" and any(ch.isdigit() for ch in candidate)
                ),
                "",
            )
            price_parts = price_blob.split()
            price = None
            if price_parts:
                try:
                    price = float(price_parts[0].replace(",", ""))
                except ValueError:
                    price = None
            last_30d = None
            if len(price_parts) > 1 and price_parts[1].endswith("%"):
                try:
                    last_30d = float(price_parts[1].replace("%", "")) / 100.0
                except ValueError:
                    last_30d = None
            rows.append(
                {
                    "ticker": ticker,
                    "asset_name": asset_name,
                    "issuer_last_traded": pd.Timestamp(last_traded),
                    "volume": volume,
                    "disclosure_count": trades,
                    "politician_count": politicians,
                    "sector": sector if sector != "N/A" else None,
                    "price": price,
                    "last_30d_return_hint": last_30d,
                }
            )
            i += 10
        return rows

    def to_frame(self, trades: list[TradeDisclosure]) -> pd.DataFrame:
        # Convert typed trade objects into a DataFrame for vectorized scoring.
        if not trades:
            return pd.DataFrame()
        rows = []
        for trade in trades:
            rows.append(
                {
                    "politician": trade.politician,
                    "ticker": trade.ticker or None,
                    "asset_name": trade.asset_name,
                    "traded_at": pd.Timestamp(trade.traded_at),
                    "published_at": pd.Timestamp(trade.published_at),
                    "transaction_type": trade.transaction_type,
                    "size_low": trade.size_low,
                    "size_high": trade.size_high,
                    "notional_mid": trade.notional_mid,
                    "signed_notional": trade.signed_notional,
                    "filing_delay_days": trade.filing_delay_days,
                    "party": trade.party,
                    "chamber": trade.chamber,
                    "state": trade.state,
                }
            )
        return pd.DataFrame(rows)
