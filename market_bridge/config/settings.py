"""
Market Bridge — Configuration & Constants
==========================================
Central config for SEC EDGAR, FRED, Polygon, and Anthropic API settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# SEC EDGAR Configuration
# ─────────────────────────────────────────────
SEC_EDGAR_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# EDGAR requires a User-Agent header with contact info
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "MarketBridge/1.0 (contact@marketbridge.ai)")

# Filing type mappings
FILING_TYPES = {
    "earnings_transcript": "8-K",
    "annual_report": "10-K",
    "quarterly_report": "10-Q",
}

# 10-K Template Sections (Item references per SEC regulation S-K)
TEMPLATE_10K_SECTIONS = {
    "business": "Item 1",
    "risk_factors": "Item 1A",
    "properties": "Item 2",
    "legal_proceedings": "Item 3",
    "mda": "Item 7",            # Management Discussion & Analysis
    "quant_disclosures": "Item 7A",
    "financial_statements": "Item 8",
    "controls": "Item 9A",
}

# 10-Q Template Sections
TEMPLATE_10Q_SECTIONS = {
    "financial_statements": "Part I, Item 1",
    "mda": "Part I, Item 2",
    "quant_disclosures": "Part I, Item 3",
    "controls": "Part I, Item 4",
    "legal_proceedings": "Part II, Item 1",
    "risk_factors": "Part II, Item 1A",
}

# 8-K Items (earnings-relevant)
ITEMS_8K_EARNINGS = [
    "Item 2.02",   # Results of Operations and Financial Condition
    "Item 7.01",   # Regulation FD Disclosure
    "Item 9.01",   # Financial Statements and Exhibits
]


# ─────────────────────────────────────────────
# FRED Configuration
# ─────────────────────────────────────────────
FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Core macro series for regime context
FRED_MACRO_SERIES = {
    "GDP":           "GDP",
    "CPI":           "CPIAUCSL",
    "UNEMPLOYMENT":  "UNRATE",
    "FED_FUNDS":     "FEDFUNDS",
    "YIELD_10Y":     "DGS10",
    "YIELD_2Y":      "DGS2",
    "YIELD_SPREAD":  "T10Y2Y",
    "VIX":           "VIXCLS",
    "CREDIT_SPREAD": "BAMLC0A0CM",
    "M2":            "M2SL",
    "INDUSTRIAL_PROD": "INDPRO",
    "PMI":           "MANEMP",
}


# ─────────────────────────────────────────────
# Polygon.io Configuration
# ─────────────────────────────────────────────
POLYGON_API_BASE = "https://api.polygon.io"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")


# ─────────────────────────────────────────────
# Anthropic / Claude API Configuration
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_MAX_TOKENS = 4096


# ─────────────────────────────────────────────
# Chunking Configuration
# ─────────────────────────────────────────────
@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategies."""
    
    # Semantic chunking (for 8-K earnings transcripts)
    semantic_model: str = "all-MiniLM-L6-v2"
    semantic_similarity_threshold: float = 0.75
    semantic_min_chunk_size: int = 200       # chars
    semantic_max_chunk_size: int = 2000      # chars
    semantic_overlap: int = 100              # chars
    
    # Template chunking (for 10-K / 10-Q)
    template_fallback_chunk_size: int = 1500
    template_overlap: int = 150


# ─────────────────────────────────────────────
# Synthesis Prompt Templates
# ─────────────────────────────────────────────
SYNTHESIS_SYSTEM_PROMPT = """You are Market Bridge, an institutional-grade financial research analyst.
You synthesize SEC filings, earnings transcripts, and macroeconomic data into
structured investment intelligence.

Your analysis must be:
- Quantitative: cite specific numbers, ratios, growth rates
- Comparative: reference prior periods, peer benchmarks
- Forward-looking: identify catalysts, risks, inflection points
- Actionable: produce clear signal (bullish / bearish / neutral) with conviction level

Output format: structured JSON matching the requested schema.
"""

EARNINGS_ANALYSIS_PROMPT = """Analyze this earnings transcript for {ticker} ({company_name}).

TRANSCRIPT CHUNKS:
{chunks}

MACRO CONTEXT:
{macro_context}

Produce a JSON response with this schema:
{{
  "ticker": str,
  "quarter": str,
  "revenue_surprise": str,
  "eps_surprise": str,
  "guidance": {{
    "direction": "raised|maintained|lowered|withdrawn",
    "details": str
  }},
  "key_themes": [str],
  "management_tone": "confident|cautious|defensive|optimistic",
  "risk_flags": [str],
  "catalyst_triggers": [str],
  "signal": "bullish|bearish|neutral",
  "conviction": "high|medium|low",
  "summary": str
}}
"""

FILING_ANALYSIS_PROMPT = """Analyze this {filing_type} filing for {ticker} ({company_name}).

FILING SECTIONS:
{sections}

MACRO CONTEXT:
{macro_context}

Produce a JSON response with this schema:
{{
  "ticker": str,
  "filing_type": str,
  "period": str,
  "financial_health": {{
    "revenue_trend": str,
    "margin_trajectory": str,
    "debt_profile": str,
    "cash_flow_quality": str
  }},
  "risk_assessment": {{
    "key_risks": [str],
    "risk_changes_from_prior": [str],
    "litigation_exposure": str
  }},
  "strategic_signals": {{
    "growth_initiatives": [str],
    "capex_direction": str,
    "m_and_a_signals": str
  }},
  "mda_insights": {{
    "management_focus_areas": [str],
    "tone_shift": str,
    "forward_guidance_clues": [str]
  }},
  "signal": "bullish|bearish|neutral",
  "conviction": "high|medium|low",
  "summary": str
}}
"""


# ─────────────────────────────────────────────
# Application Defaults
# ─────────────────────────────────────────────
@dataclass
class AppConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    cache_dir: str = os.path.expanduser("~/.market_bridge/cache")
    log_level: str = "INFO"
    max_concurrent_requests: int = 5
    request_timeout: int = 30
