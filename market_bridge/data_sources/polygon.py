"""
Market Bridge — Polygon.io Data Source
======================================
Fetches market data (price bars, ticker details, financials) from Polygon.io
to enrich LLM context with real-time and historical market information.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from market_bridge.config.settings import POLYGON_API_BASE, POLYGON_API_KEY
from market_bridge.utils.helpers import get_logger, build_session

logger = get_logger("polygon")


class PolygonClient:
    """Client for the Polygon.io REST API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or POLYGON_API_KEY
        self.session = build_session()
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not set — market data unavailable.")
    
    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{POLYGON_API_BASE}{path}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    
    # ── Ticker Details ──────────────────────────
    def get_ticker_details(self, ticker: str) -> Dict:
        """Get company details for a ticker."""
        if not self.api_key:
            return {}
        data = self._get(f"/v3/reference/tickers/{ticker.upper()}")
        results = data.get("results", {})
        return {
            "ticker": results.get("ticker"),
            "name": results.get("name"),
            "market_cap": results.get("market_cap"),
            "sic_code": results.get("sic_code"),
            "sic_description": results.get("sic_description"),
            "sector": results.get("type"),
            "homepage": results.get("homepage_url"),
            "total_employees": results.get("total_employees"),
            "description": results.get("description"),
        }
    
    # ── Price Data ──────────────────────────────
    def get_price_bars(
        self,
        ticker: str,
        timespan: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 60,
    ) -> List[Dict]:
        """Fetch OHLCV bars for a ticker."""
        if not self.api_key:
            return []
        
        to_dt = to_date or datetime.now().strftime("%Y-%m-%d")
        from_dt = from_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        data = self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/{timespan}/{from_dt}/{to_dt}",
            params={"adjusted": "true", "sort": "desc", "limit": limit},
        )
        
        results = data.get("results", [])
        return [
            {
                "date": datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d"),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw"),
            }
            for bar in results
        ]
    
    # ── Stock Financials ────────────────────────
    def get_financials(
        self,
        ticker: str,
        period: str = "quarterly",
        limit: int = 4,
    ) -> List[Dict]:
        """Fetch standardized financial statements from Polygon."""
        if not self.api_key:
            return []
        
        data = self._get(
            f"/vX/reference/financials",
            params={
                "ticker": ticker.upper(),
                "timeframe": period,
                "limit": limit,
                "sort": "filing_date",
                "order": "desc",
            },
        )
        return data.get("results", [])
    
    # ── Format Market Context ───────────────────
    def format_market_context(self, ticker: str) -> str:
        """Build a market context block for LLM prompts."""
        lines = [f"=== MARKET CONTEXT ({ticker.upper()}) ==="]
        
        # Company details
        details = self.get_ticker_details(ticker)
        if details.get("name"):
            lines.append(f"  Company: {details['name']}")
        if details.get("market_cap"):
            mc = details["market_cap"]
            if mc >= 1e12:
                lines.append(f"  Market Cap: ${mc/1e12:.1f}T")
            elif mc >= 1e9:
                lines.append(f"  Market Cap: ${mc/1e9:.1f}B")
            else:
                lines.append(f"  Market Cap: ${mc/1e6:.0f}M")
        if details.get("sic_description"):
            lines.append(f"  Sector: {details['sic_description']}")
        if details.get("total_employees"):
            lines.append(f"  Employees: {details['total_employees']:,}")
        
        # Recent price action
        bars = self.get_price_bars(ticker, limit=20)
        if bars:
            latest = bars[0]
            oldest = bars[-1]
            price_chg = ((latest["close"] - oldest["close"]) / oldest["close"]) * 100
            lines.append(f"\n  Latest Close: ${latest['close']:.2f} ({latest['date']})")
            lines.append(f"  20-Day Change: {price_chg:+.1f}%")
            
            avg_vol = sum(b["volume"] for b in bars) / len(bars)
            lines.append(f"  Avg Volume (20d): {avg_vol:,.0f}")
            
            high_20 = max(b["high"] for b in bars)
            low_20 = min(b["low"] for b in bars)
            lines.append(f"  20-Day Range: ${low_20:.2f} — ${high_20:.2f}")
        
        return "\n".join(lines)
