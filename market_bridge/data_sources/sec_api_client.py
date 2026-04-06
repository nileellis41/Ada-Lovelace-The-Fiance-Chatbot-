"""
Market Bridge — SEC-API.io Data Source
======================================
Uses the sec-api.io Python SDK for filing discovery (QueryApi),
section extraction (ExtractorApi), XBRL financial data (XbrlApi),
and document download (RenderApi).

Key design decisions:
  - ExtractorApi handles template-based section extraction natively
    (returns clean text/HTML by Item number for 10-K, 10-Q, 8-K)
  - QueryApi uses Lucene syntax for precise filing lookup
  - XbrlApi provides structured financial statements as JSON
  - We still apply semantic chunking to 8-K earnings transcripts
    after extraction, since those are unstructured narrative text
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from market_bridge.config.settings import (
    TEMPLATE_10K_SECTIONS,
    TEMPLATE_10Q_SECTIONS,
    ITEMS_8K_EARNINGS,
)
from market_bridge.utils.helpers import get_logger, normalize_text, clean_html

logger = get_logger("sec_api")


# ─────────────────────────────────────────────
# sec-api.io section codes
# ─────────────────────────────────────────────
# 10-K section codes (used by ExtractorApi)
SECTION_CODES_10K = {
    "1":   "Business",
    "1A":  "Risk Factors",
    "1B":  "Unresolved Staff Comments",
    "2":   "Properties",
    "3":   "Legal Proceedings",
    "4":   "Mine Safety Disclosures",
    "5":   "Market for Common Equity",
    "6":   "Selected Financial Data",
    "7":   "MD&A",
    "7A":  "Quantitative & Qualitative Disclosures",
    "8":   "Financial Statements",
    "9":   "Disagreements with Accountants",
    "9A":  "Controls and Procedures",
    "9B":  "Other Information",
    "10":  "Directors & Executive Officers",
    "11":  "Executive Compensation",
    "12":  "Security Ownership",
    "13":  "Certain Relationships",
    "14":  "Principal Accountant Fees",
}

# 10-Q section codes
SECTION_CODES_10Q = {
    "part1item1":  "Financial Statements",
    "part1item2":  "MD&A",
    "part1item3":  "Quantitative & Qualitative Disclosures",
    "part1item4":  "Controls and Procedures",
    "part2item1":  "Legal Proceedings",
    "part2item1a": "Risk Factors",
    "part2item2":  "Unregistered Sales of Equity",
    "part2item3":  "Defaults Upon Senior Securities",
    "part2item4":  "Mine Safety Disclosures",
    "part2item5":  "Other Information",
    "part2item6":  "Exhibits",
}

# 8-K item codes relevant to earnings
SECTION_CODES_8K = {
    "1-1":  "Entry into Material Agreement",
    "1-2":  "Termination of Material Agreement",
    "2-2":  "Results of Operations (Earnings)",
    "7-1":  "Regulation FD Disclosure",
    "8-1":  "Other Events",
    "9-1":  "Financial Statements and Exhibits",
}

# Priority sections for analysis (most investment-relevant)
PRIORITY_SECTIONS_10K = ["1A", "7", "7A", "8", "1"]
PRIORITY_SECTIONS_10Q = ["part1item2", "part1item1", "part2item1a", "part1item3"]
PRIORITY_SECTIONS_8K  = ["2-2", "7-1", "9-1"]


@dataclass
class SecFiling:
    """Represents a filing retrieved from sec-api.io."""
    ticker: str
    company_name: str
    form_type: str
    filed_at: str
    period_of_report: str
    accession_number: str
    filing_url: str
    cik: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    xbrl_data: Optional[Dict] = None
    raw_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecApiClient:
    """
    Unified client wrapping sec-api.io SDK endpoints:
      - QueryApi:     filing discovery & search
      - ExtractorApi: template-based section extraction
      - XbrlApi:      structured financial statements
      - RenderApi:    raw document download
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEC_API_KEY", "")
        if not self.api_key:
            logger.warning("SEC_API_KEY not set. sec-api.io calls will fail.")
        
        # Lazy-init SDK classes
        self._query_api = None
        self._extractor_api = None
        self._xbrl_api = None
        self._render_api = None
    
    # ── Lazy SDK initialization ─────────────────
    @property
    def query_api(self):
        if self._query_api is None:
            from sec_api import QueryApi
            self._query_api = QueryApi(api_key=self.api_key)
        return self._query_api
    
    @property
    def extractor_api(self):
        if self._extractor_api is None:
            from sec_api import ExtractorApi
            self._extractor_api = ExtractorApi(api_key=self.api_key)
        return self._extractor_api
    
    @property
    def xbrl_api(self):
        if self._xbrl_api is None:
            from sec_api import XbrlApi
            self._xbrl_api = XbrlApi(api_key=self.api_key)
        return self._xbrl_api
    
    @property
    def render_api(self):
        if self._render_api is None:
            from sec_api import RenderApi
            self._render_api = RenderApi(api_key=self.api_key)
        return self._render_api
    
    # ═══════════════════════════════════════════
    # QUERY API — Filing Discovery
    # ═══════════════════════════════════════════
    def search_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        date_from: str = "2020-01-01",
        date_to: str = "2026-12-31",
        size: int = 5,
    ) -> List[SecFiling]:
        """
        Search for filings by ticker and form type using Lucene query syntax.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            form_type: SEC form type (e.g., "10-K", "10-Q", "8-K")
            date_from/date_to: Date range for search
            size: Number of results to return (max 50)
        
        Returns:
            List of SecFiling objects with metadata (no content yet)
        """
        query_str = (
            f'ticker:{ticker.upper()} '
            f'AND formType:"{form_type}" '
            f'AND filedAt:[{date_from} TO {date_to}]'
        )
        
        search_params = {
            "query": {"query_string": {"query": query_str}},
            "from": "0",
            "size": str(min(size, 50)),
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        
        logger.info(f"Querying sec-api.io: {query_str}")
        response = self.query_api.get_filings(search_params)
        
        filings_data = response.get("filings", []) if isinstance(response, dict) else response
        if isinstance(filings_data, dict):
            filings_data = filings_data.get("filings", [])
        
        filings = []
        for f in filings_data:
            filing = SecFiling(
                ticker=f.get("ticker", ticker.upper()),
                company_name=f.get("companyName", ""),
                form_type=f.get("formType", form_type),
                filed_at=f.get("filedAt", ""),
                period_of_report=f.get("periodOfReport", ""),
                accession_number=f.get("accessionNo", ""),
                filing_url=f.get("linkToFilingDetails", ""),
                cik=f.get("cik", ""),
                metadata={
                    "description": f.get("description", ""),
                    "items": f.get("items", []),
                    "document_format_files": f.get("documentFormatFiles", []),
                },
            )
            filings.append(filing)
        
        logger.info(f"Found {len(filings)} {form_type} filings for {ticker}")
        return filings
    
    def get_latest_filing(self, ticker: str, form_type: str = "10-K") -> Optional[SecFiling]:
        """Get the single most recent filing of a given type."""
        filings = self.search_filings(ticker, form_type=form_type, size=1)
        return filings[0] if filings else None
    
    def get_latest_earnings_8k(self, ticker: str, count: int = 1) -> List[SecFiling]:
        """
        Get recent earnings-related 8-K filings.
        Filters for Item 2.02 (Results of Operations).
        """
        # Pull more than needed and filter by earnings items
        all_8ks = self.search_filings(ticker, form_type="8-K", size=20)
        
        earnings_filings = []
        for f in all_8ks:
            items = f.metadata.get("items", [])
            # Check for Item 2.02 (earnings) or 7.01 (Reg FD — often earnings)
            if any(item in items for item in ["2.02", "7.01"]):
                earnings_filings.append(f)
            if len(earnings_filings) >= count:
                break
        
        logger.info(f"Found {len(earnings_filings)} earnings 8-K(s) for {ticker}")
        return earnings_filings
    
    # ═══════════════════════════════════════════
    # EXTRACTOR API — Template Section Extraction
    # ═══════════════════════════════════════════
    def extract_section(
        self,
        filing_url: str,
        section_code: str,
        output_type: str = "text",
    ) -> str:
        """
        Extract a specific section from a filing using sec-api Extractor.
        
        Args:
            filing_url: URL to the filing (from linkToFilingDetails)
            section_code: Section identifier (e.g., "7" for MD&A in 10-K,
                         "part1item2" for MD&A in 10-Q, "2-2" for earnings 8-K)
            output_type: "text" for clean text, "html" for standardized HTML
        
        Returns:
            Extracted section content as string
        """
        try:
            content = self.extractor_api.get_section(
                filing_url, section_code, output_type
            )
            if output_type == "html" and content:
                content = clean_html(content)
            return normalize_text(content) if content else ""
        except Exception as e:
            logger.warning(f"Failed to extract section {section_code} from {filing_url}: {e}")
            return ""
    
    def extract_all_sections(
        self,
        filing: SecFiling,
        priority_only: bool = True,
        output_type: str = "text",
    ) -> Dict[str, str]:
        """
        Extract all (or priority) sections from a filing.
        
        For 10-K: Items 1, 1A, 7, 7A, 8 (priority) or all Items 1-14
        For 10-Q: Part I Items 1-4, Part II Items 1-6
        For 8-K:  Items 2.02, 7.01, 9.01 (earnings-relevant)
        
        Returns:
            Dict mapping section labels to extracted text content
        """
        form = filing.form_type.upper()
        
        if "10-K" in form:
            codes = PRIORITY_SECTIONS_10K if priority_only else list(SECTION_CODES_10K.keys())
            labels = SECTION_CODES_10K
        elif "10-Q" in form:
            codes = PRIORITY_SECTIONS_10Q if priority_only else list(SECTION_CODES_10Q.keys())
            labels = SECTION_CODES_10Q
        elif "8-K" in form:
            codes = PRIORITY_SECTIONS_8K if priority_only else list(SECTION_CODES_8K.keys())
            labels = SECTION_CODES_8K
        else:
            logger.warning(f"Unsupported form type for extraction: {form}")
            return {}
        
        sections = {}
        for code in codes:
            label = labels.get(code, f"Section {code}")
            full_label = f"Item {code} — {label}"
            
            logger.info(f"Extracting {full_label} from {filing.ticker} {form}")
            content = self.extract_section(filing.filing_url, code, output_type)
            
            if content and len(content) > 50:
                sections[full_label] = content
                logger.info(f"  ✓ {full_label}: {len(content)} chars")
            else:
                logger.debug(f"  ✗ {full_label}: empty or too short")
        
        filing.sections = sections
        return sections
    
    # ═══════════════════════════════════════════
    # XBRL API — Structured Financial Data
    # ═══════════════════════════════════════════
    def get_xbrl_financials(self, filing: SecFiling) -> Optional[Dict]:
        """
        Convert XBRL data from a 10-K/10-Q filing to structured JSON.
        Returns income statement, balance sheet, cash flow statement.
        """
        if not filing.filing_url:
            return None
        
        try:
            xbrl_json = self.xbrl_api.xbrl_to_json(htm_url=filing.filing_url)
            
            if isinstance(xbrl_json, str):
                xbrl_json = json.loads(xbrl_json)
            
            filing.xbrl_data = xbrl_json
            
            # Extract key financial statement sections
            financials = {
                "income_statement": xbrl_json.get("StatementsOfIncome", {}),
                "balance_sheet": xbrl_json.get("BalanceSheets", {}),
                "cash_flow": xbrl_json.get("StatementsOfCashFlows", {}),
                "cover_page": xbrl_json.get("CoverPage", {}),
            }
            
            logger.info(f"XBRL extracted for {filing.ticker}: "
                        f"IS={bool(financials['income_statement'])}, "
                        f"BS={bool(financials['balance_sheet'])}, "
                        f"CF={bool(financials['cash_flow'])}")
            
            return financials
        except Exception as e:
            logger.warning(f"XBRL extraction failed for {filing.filing_url}: {e}")
            return None
    
    def format_xbrl_context(self, xbrl_data: Dict) -> str:
        """Format XBRL financial data into readable context for LLM prompts."""
        lines = ["=== STRUCTURED FINANCIAL DATA (XBRL) ==="]
        
        for statement_name, data in xbrl_data.items():
            if not data or statement_name == "cover_page":
                continue
            
            lines.append(f"\n  [{statement_name.upper().replace('_', ' ')}]")
            
            if isinstance(data, dict):
                for metric, values in list(data.items())[:15]:
                    # Clean up XBRL tag names
                    clean_name = metric.replace("us-gaap_", "").replace("_", " ")
                    if isinstance(values, list) and values:
                        latest = values[0]
                        val = latest.get("value", "N/A")
                        period = latest.get("period", {})
                        lines.append(f"    {clean_name}: {val}")
                    elif isinstance(values, (int, float, str)):
                        lines.append(f"    {clean_name}: {values}")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════
    # RENDER API — Raw Document Download
    # ═══════════════════════════════════════════
    def download_filing(self, filing_url: str) -> str:
        """Download the raw filing content."""
        try:
            content = self.render_api.get_filing(filing_url)
            return normalize_text(clean_html(content)) if content else ""
        except Exception as e:
            logger.warning(f"Failed to download filing from {filing_url}: {e}")
            return ""
    
    # ═══════════════════════════════════════════
    # CONVENIENCE — Full Pipeline Methods
    # ═══════════════════════════════════════════
    def pull_annual_report(
        self,
        ticker: str,
        with_xbrl: bool = True,
        priority_only: bool = True,
    ) -> Optional[SecFiling]:
        """
        Full pipeline: find latest 10-K → extract sections → get XBRL data.
        """
        filing = self.get_latest_filing(ticker, form_type="10-K")
        if not filing:
            logger.error(f"No 10-K found for {ticker}")
            return None
        
        self.extract_all_sections(filing, priority_only=priority_only)
        
        if with_xbrl:
            xbrl = self.get_xbrl_financials(filing)
            if xbrl:
                filing.metadata["xbrl_financials"] = xbrl
        
        return filing
    
    def pull_quarterly_report(
        self,
        ticker: str,
        with_xbrl: bool = True,
        priority_only: bool = True,
    ) -> Optional[SecFiling]:
        """
        Full pipeline: find latest 10-Q → extract sections → get XBRL data.
        """
        filing = self.get_latest_filing(ticker, form_type="10-Q")
        if not filing:
            logger.error(f"No 10-Q found for {ticker}")
            return None
        
        self.extract_all_sections(filing, priority_only=priority_only)
        
        if with_xbrl:
            xbrl = self.get_xbrl_financials(filing)
            if xbrl:
                filing.metadata["xbrl_financials"] = xbrl
        
        return filing
    
    def pull_earnings_transcript(self, ticker: str) -> Optional[SecFiling]:
        """
        Full pipeline: find latest earnings 8-K → extract Item 2.02 content.
        This returns raw narrative text suitable for semantic chunking.
        """
        filings = self.get_latest_earnings_8k(ticker, count=1)
        if not filings:
            logger.error(f"No earnings 8-K found for {ticker}")
            return None
        
        filing = filings[0]
        self.extract_all_sections(filing, priority_only=True)
        
        # If Extractor didn't return content, fall back to raw download
        if not filing.sections:
            logger.info(f"Extractor returned empty — downloading raw 8-K for {ticker}")
            raw = self.download_filing(filing.filing_url)
            if raw:
                filing.raw_text = raw
                filing.sections = {"Item 2.02 — Results of Operations (raw)": raw}
        
        return filing
