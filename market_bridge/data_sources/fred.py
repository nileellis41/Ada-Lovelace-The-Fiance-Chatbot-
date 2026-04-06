"""
Market Bridge — FRED Data Source
================================
Pulls macroeconomic time-series from FRED API to provide regime context
for LLM synthesis (yield curve, VIX, credit spreads, GDP, etc.).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from market_bridge.config.settings import FRED_API_BASE, FRED_API_KEY, FRED_MACRO_SERIES
from market_bridge.utils.helpers import get_logger, build_session

logger = get_logger("fred")


class FredClient:
    """Client for the FRED (Federal Reserve Economic Data) API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or FRED_API_KEY
        self.session = build_session()
        if not self.api_key:
            logger.warning("FRED_API_KEY not set — macro context will be unavailable.")
    
    def _get(self, endpoint: str, params: Dict) -> Dict:
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        resp = self.session.get(f"{FRED_API_BASE}/{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()
    
    def get_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict]:
        """Fetch observations for a FRED series."""
        if not self.api_key:
            return []
        
        params = {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": limit,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        
        data = self._get("series/observations", params)
        observations = data.get("observations", [])
        
        return [
            {"date": obs["date"], "value": float(obs["value"])}
            for obs in observations
            if obs["value"] != "."
        ]
    
    def get_latest_value(self, series_id: str) -> Optional[Dict]:
        """Get the most recent observation for a series."""
        obs = self.get_series(series_id, limit=1)
        return obs[0] if obs else None
    
    def get_macro_snapshot(self) -> Dict[str, Optional[Dict]]:
        """Pull the latest value for all core macro series."""
        snapshot = {}
        for label, series_id in FRED_MACRO_SERIES.items():
            try:
                val = self.get_latest_value(series_id)
                snapshot[label] = val
                logger.debug(f"FRED {label}: {val}")
            except Exception as e:
                logger.warning(f"Failed to fetch FRED series {label} ({series_id}): {e}")
                snapshot[label] = None
        return snapshot
    
    def format_macro_context(self, snapshot: Optional[Dict] = None) -> str:
        """Format macro snapshot into a readable context block for LLM prompts."""
        if snapshot is None:
            snapshot = self.get_macro_snapshot()
        
        lines = ["=== MACROECONOMIC CONTEXT (FRED) ==="]
        for label, data in snapshot.items():
            if data:
                lines.append(f"  {label}: {data['value']} (as of {data['date']})")
            else:
                lines.append(f"  {label}: N/A")
        
        # Derived signals
        spread = snapshot.get("YIELD_SPREAD")
        vix = snapshot.get("VIX")
        if spread and spread.get("value") is not None:
            val = spread["value"]
            if val < 0:
                lines.append(f"\n  >> SIGNAL: Yield curve INVERTED ({val}bps) — recession risk elevated")
            elif val < 0.5:
                lines.append(f"\n  >> SIGNAL: Yield curve FLAT ({val}bps) — late-cycle dynamics")
            else:
                lines.append(f"\n  >> SIGNAL: Yield curve NORMAL ({val}bps)")
        
        if vix and vix.get("value") is not None:
            vix_val = vix["value"]
            if vix_val > 30:
                lines.append(f"  >> SIGNAL: VIX ELEVATED ({vix_val}) — high volatility regime")
            elif vix_val > 20:
                lines.append(f"  >> SIGNAL: VIX MODERATE ({vix_val}) — transitional regime")
            else:
                lines.append(f"  >> SIGNAL: VIX LOW ({vix_val}) — risk-on regime")
        
        return "\n".join(lines)
