"""
Market Bridge — Utilities
=========================
Shared helpers: async HTTP client, text normalization, logging.
"""

import re
import html
import logging
import time
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from market_bridge.config.settings import SEC_USER_AGENT


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(f"market_bridge.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ─────────────────────────────────────────────
# HTTP Session with retry + rate-limit respect
# ─────────────────────────────────────────────
def build_session(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    user_agent: str = SEC_USER_AGENT,
) -> requests.Session:
    """Build a requests Session with automatic retries and SEC-compliant headers."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


class RateLimiter:
    """Simple token-bucket rate limiter for SEC EDGAR (10 req/sec)."""
    
    def __init__(self, calls_per_second: float = 10.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# Global rate limiter for SEC
sec_rate_limiter = RateLimiter(calls_per_second=9.0)  # stay under 10/s


# ─────────────────────────────────────────────
# Text Cleaning & Normalization
# ─────────────────────────────────────────────
def clean_html(raw_html: str) -> str:
    """Strip HTML tags and decode entities, preserving paragraph structure."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    # Convert <br>, <p>, <div> to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|li|h\d)>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Standard text normalization for chunking input."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\xa0", " ", text)           # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 consecutive newlines
    return text.strip()


def extract_tables_from_html(html_content: str) -> list:
    """Extract tables from HTML content as list of dicts."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables
