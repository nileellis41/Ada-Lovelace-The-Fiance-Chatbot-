"""
Polygon.io Market Data Client
──────────────────────────────
Real-time and historical price data, ticker details, and news.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config import Config

logger = logging.getLogger(__name__)


class PolygonClient:
    BASE = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.POLYGON_API_KEY
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not set — Polygon queries will fail")

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        params = params or {}
        params["apiKey"] = self.api_key
        resp = requests.get(f"{self.BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Price data ───────────────────────────────────────
    def get_bars(
        self,
        ticker: str,
        timespan: str = "day",
        multiplier: int = 1,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 120,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars."""
        if not start:
            start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")

        path = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{start}/{end}"
        data = self._get(path, {"limit": limit, "sort": "desc"})

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def get_previous_close(self, ticker: str) -> Optional[dict]:
        """Get previous day's close data."""
        try:
            data = self._get(f"/v2/aggs/ticker/{ticker.upper()}/prev")
            results = data.get("results", [])
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Polygon prev close error: {e}")
            return None

    # ── Ticker info ──────────────────────────────────────
    def get_ticker_details(self, ticker: str) -> dict:
        """Get company details for a ticker."""
        try:
            data = self._get(f"/v3/reference/tickers/{ticker.upper()}")
            return data.get("results", {})
        except Exception as e:
            logger.error(f"Polygon ticker details error: {e}")
            return {}

    # ── News ─────────────────────────────────────────────
    def get_news(self, ticker: Optional[str] = None, limit: int = 5) -> list[dict]:
        """Get recent market news, optionally filtered by ticker."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker.upper()
        try:
            data = self._get("/v2/reference/news", params)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Polygon news error: {e}")
            return []

    # ── Formatted output for LLM context ─────────────────
    def format_quote_text(self, ticker: str) -> str:
        """Human-readable quote for LLM context injection."""
        prev = self.get_previous_close(ticker)
        if not prev:
            return f"No price data found for {ticker}."

        details = self.get_ticker_details(ticker)
        name = details.get("name", ticker.upper())

        return (
            f"═══ {name} ({ticker.upper()}) ═══\n"
            f"  Close: ${prev.get('c', 'N/A'):.2f}\n"
            f"  Open:  ${prev.get('o', 'N/A'):.2f}\n"
            f"  High:  ${prev.get('h', 'N/A'):.2f}\n"
            f"  Low:   ${prev.get('l', 'N/A'):.2f}\n"
            f"  Volume: {prev.get('v', 'N/A'):,.0f}"
        )

    def format_news_text(self, ticker: Optional[str] = None, limit: int = 5) -> str:
        """Formatted news digest for LLM context."""
        articles = self.get_news(ticker, limit)
        if not articles:
            return "No recent news found."

        lines = [f"═══ MARKET NEWS {'(' + ticker.upper() + ')' if ticker else ''} ═══"]
        for a in articles:
            title = a.get("title", "Untitled")
            pub = a.get("published_utc", "")[:10]
            source = a.get("publisher", {}).get("name", "Unknown")
            lines.append(f"  [{pub}] {title} — {source}")
        return "\n".join(lines)
