"""
SEC EDGAR Client
────────────────
Fetch and parse 10-K / 10-Q filings via the EDGAR full-text search API.
"""
import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import Config

logger = logging.getLogger(__name__)


class EDGARClient:
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    FILING_BASE = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or Config.EDGAR_USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    # ── Company lookup ───────────────────────────────────
    def search_company(self, query: str) -> list[dict]:
        """Search EDGAR company tickers/names."""
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {"q": query, "dateRange": "custom", "startdt": "2020-01-01"}
        try:
            resp = self.session.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("hits", {}).get("hits", [])
        except Exception:
            # Fallback: use the full-text search API
            return self._fulltext_search(query)

    def _fulltext_search(self, query: str, forms: str = "10-K,10-Q", limit: int = 5) -> list[dict]:
        """EDGAR full-text search (EFTS)."""
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": query,
            "forms": forms,
            "dateRange": "custom",
            "startdt": "2022-01-01",
        }
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("hits", {}).get("hits", [])[:limit]
        except Exception as e:
            logger.error(f"EDGAR full-text search error: {e}")
            return []

    # ── Filing retrieval ─────────────────────────────────
    def get_recent_filings(
        self, ticker: str, form_type: str = "10-K", count: int = 3
    ) -> list[dict]:
        """Get recent filings for a ticker via the submissions API."""
        # First resolve ticker → CIK
        cik = self._resolve_cik(ticker)
        if not cik:
            return []

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"EDGAR submissions error: {e}")
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type and len(results) < count:
                acc_clean = accessions[i].replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik}/{acc_clean}/{docs[i]}"
                )
                results.append({
                    "form": form,
                    "date": dates[i],
                    "accession": accessions[i],
                    "url": filing_url,
                })
        return results

    def fetch_filing_text(self, url: str, max_chars: int = 50_000) -> str:
        """Download and extract text from a filing URL."""
        time.sleep(0.2)  # SEC rate limiting courtesy
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            # Clean excessive whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text[:max_chars]
        except Exception as e:
            logger.error(f"EDGAR filing fetch error: {e}")
            return ""

    # ── Internal helpers ─────────────────────────────────
    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """Resolve a ticker symbol to a zero-padded CIK."""
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "company": ticker,
            "type": "",
            "dateb": "",
            "owner": "include",
            "count": "1",
            "search_text": "",
            "action": "getcompany",
            "company": "",
            "CIK": ticker,
            "type": "10-K",
            "dateb": "",
            "owner": "include",
            "count": "1",
            "output": "atom",
        }
        try:
            # Use the company tickers JSON (faster)
            resp = self.session.get(
                "https://www.sec.gov/files/company_tickers.json", timeout=10
            )
            resp.raise_for_status()
            tickers = resp.json()
            for entry in tickers.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        except Exception as e:
            logger.error(f"CIK resolution error: {e}")
        return None

    def format_filings_text(self, ticker: str, form_type: str = "10-K") -> str:
        """Get a formatted summary of recent filings for LLM context."""
        filings = self.get_recent_filings(ticker, form_type)
        if not filings:
            return f"No {form_type} filings found for {ticker}."

        lines = [f"═══ EDGAR {form_type} FILINGS: {ticker.upper()} ═══"]
        for f in filings:
            lines.append(f"  [{f['date']}] {f['form']} — {f['url']}")
        return "\n".join(lines)
