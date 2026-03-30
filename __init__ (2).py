"""
FRED (Federal Reserve Economic Data) Client
─────────────────────────────────────────────
Pulls macro series: GDP, CPI, Fed Funds, unemployment, yield curves, etc.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config import Config

logger = logging.getLogger(__name__)

# ── Common macro series ──────────────────────────────────
MACRO_SERIES = {
    "fed_funds":    ("DFF",      "Federal Funds Effective Rate"),
    "gdp":          ("GDP",      "Gross Domestic Product"),
    "gdp_growth":   ("A191RL1Q225SBEA", "Real GDP Growth Rate"),
    "cpi":          ("CPIAUCSL", "Consumer Price Index"),
    "core_cpi":     ("CPILFESL", "Core CPI (ex Food & Energy)"),
    "unemployment": ("UNRATE",   "Unemployment Rate"),
    "t10y2y":       ("T10Y2Y",   "10Y-2Y Treasury Spread"),
    "t10y3m":       ("T10Y3M",   "10Y-3M Treasury Spread"),
    "dgs10":        ("DGS10",    "10-Year Treasury Yield"),
    "dgs2":         ("DGS2",     "2-Year Treasury Yield"),
    "vix":          ("VIXCLS",   "CBOE Volatility Index"),
    "m2":           ("M2SL",     "M2 Money Supply"),
    "initial_claims": ("ICSA",   "Initial Jobless Claims"),
    "nfp":          ("PAYEMS",   "Total Nonfarm Payrolls"),
    "pce":          ("PCEPI",    "PCE Price Index"),
    "housing_starts": ("HOUST",  "Housing Starts"),
    "retail_sales": ("RSAFS",    "Advance Retail Sales"),
    "ism_mfg":      ("MANEMP",   "Manufacturing Employment (ISM proxy)"),
}


class FREDClient:
    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.FRED_API_KEY
        if not self.api_key:
            logger.warning("FRED_API_KEY not set — FRED queries will fail")

    # ── Core fetch ───────────────────────────────────────
    def get_series(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch a FRED series as a DataFrame."""
        if not start:
            start = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "desc",
            "limit": limit,
        }
        resp = requests.get(f"{self.BASE_URL}/series/observations", params=params, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])

        df = pd.DataFrame(obs)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna().reset_index(drop=True)

    def get_series_info(self, series_id: str) -> dict:
        """Get metadata for a FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        resp = requests.get(f"{self.BASE_URL}/series", params=params, timeout=10)
        resp.raise_for_status()
        serieses = resp.json().get("seriess", [])
        return serieses[0] if serieses else {}

    # ── Convenience helpers ──────────────────────────────
    def latest_value(self, series_id: str) -> Optional[float]:
        """Return the most recent observation."""
        df = self.get_series(series_id, limit=1)
        if df.empty:
            return None
        return float(df.iloc[0]["value"])

    def macro_snapshot(self) -> dict:
        """Pull latest values for core macro indicators."""
        snapshot = {}
        for key, (sid, label) in MACRO_SERIES.items():
            try:
                val = self.latest_value(sid)
                snapshot[key] = {"series_id": sid, "label": label, "value": val}
            except Exception as e:
                logger.error(f"FRED snapshot error for {sid}: {e}")
                snapshot[key] = {"series_id": sid, "label": label, "value": None}
        return snapshot

    def format_snapshot_text(self) -> str:
        """Human-readable macro snapshot for LLM context injection."""
        snap = self.macro_snapshot()
        lines = ["═══ MACRO SNAPSHOT (FRED) ═══"]
        for key, info in snap.items():
            val = info["value"]
            val_str = f"{val:.2f}" if val is not None else "N/A"
            lines.append(f"  {info['label']}: {val_str}")
        return "\n".join(lines)

    def search_series(self, query: str, limit: int = 10) -> list[dict]:
        """Search FRED for series matching a query string."""
        params = {
            "search_text": query,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
        }
        resp = requests.get(f"{self.BASE_URL}/series/search", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("seriess", [])
