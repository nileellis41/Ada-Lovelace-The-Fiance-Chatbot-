"""
Finance Agent
──────────────
Orchestrates Claude API, FRED, EDGAR, Polygon, and RAG
to answer finance questions with grounded context.
"""
import json
import logging
import re
from typing import Generator, Optional

import anthropic

from config import Config
from data_sources import FREDClient, MACRO_SERIES
from data_sources.edgar_client import EDGARClient
from data_sources.polygon_client import PolygonClient
from rag import VectorStore

logger = logging.getLogger(__name__)

# ── Intent detection patterns ────────────────────────────
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
MACRO_KEYWORDS = {
    "fed funds", "interest rate", "gdp", "inflation", "cpi", "unemployment",
    "yield curve", "treasury", "money supply", "m2", "jobless claims",
    "nonfarm", "payroll", "pce", "housing starts", "retail sales", "vix",
    "macro", "economy", "recession", "rate cut", "rate hike", "fomc",
}
FILING_KEYWORDS = {"10-k", "10-q", "annual report", "quarterly report", "sec filing", "edgar", "filing"}
PRICE_KEYWORDS = {"price", "quote", "close", "stock", "share", "market cap", "trading"}
NEWS_KEYWORDS = {"news", "headline", "latest", "recent", "today", "breaking"}

# ── Deep analysis intent patterns ────────────────────────
DEEP_ANALYSIS_KEYWORDS = {
    "analyze", "analysis", "deep dive", "research", "synthesize",
    "investment thesis", "bull case", "bear case", "due diligence",
    "what's the signal", "earnings analysis", "filing analysis",
}

EARNINGS_KEYWORDS = {
    "earnings", "earnings call", "transcript", "results of operations",
    "revenue surprise", "eps surprise", "guidance", "8-k", "8k",
    "quarterly results", "beat", "miss", "management tone",
    "earnings report",
}

ANNUAL_REPORT_KEYWORDS = {
    "10-k", "10k", "annual report", "annual filing", "risk factors",
    "md&a", "management discussion", "financial statements",
    "annual analysis",
}

QUARTERLY_REPORT_KEYWORDS = {
    "10-q", "10q", "quarterly report", "quarterly filing",
    "quarterly analysis",
}


SYSTEM_PROMPT = """\
You are Ada Lovelace — a senior quantitative finance analyst and AI-powered research
assistant. You operate at the intersection of three intelligence layers:

═══ LAYER 1: LIVE DATA PIPELINES ═══
You have real-time access to:
- FRED API: 18 macroeconomic indicators (Fed Funds, CPI, GDP, yield curve, VIX,
  credit spreads, M2, industrial production, PMI, unemployment, etc.)
- Polygon.io: Live price quotes, OHLCV bars, company details, market cap, news
- SEC EDGAR via sec-api.io: Full-text search, filing metadata, XBRL financials

═══ LAYER 2: RAG KNOWLEDGE BASE ═══
You maintain a ChromaDB vector store of ingested financial documents — SEC filings,
research reports, earnings transcripts, and custom uploads. You retrieve the most
semantically relevant chunks for each query using cosine similarity search.

═══ LAYER 3: MARKET BRIDGE DEEP ANALYSIS ═══
For in-depth company research, you can trigger a structured analysis pipeline:

  EARNINGS ANALYSIS (8-K):
    sec-api.io ExtractorApi → Item 2.02 (Results of Operations) extraction
    → Semantic chunking (sentence-transformer embeddings detect topic boundaries
      in unstructured earnings narrative) → Claude synthesis → structured JSON
      with: revenue/EPS surprise, guidance direction, management tone, key themes,
      catalyst triggers, risk flags, and a bullish/bearish/neutral signal with
      conviction level.

  ANNUAL REPORT ANALYSIS (10-K):
    sec-api.io ExtractorApi → Template-based section extraction by Item number
    (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 7A Quantitative
    Disclosures, Item 8 Financial Statements) → Template chunking with paragraph-
    boundary splits and configurable overlap → Claude synthesis → structured JSON
    with: financial health (revenue trend, margin trajectory, debt profile, cash
    flow quality), risk assessment (key risks, litigation exposure, changes from
    prior), strategic signals (growth initiatives, CapEx direction, M&A signals),
    MD&A insights (management focus areas, tone shift, forward guidance clues),
    and a bullish/bearish/neutral signal with conviction level.

  QUARTERLY REPORT ANALYSIS (10-Q):
    Same template chunking pipeline as 10-K but targeting Part I/Part II items
    (MD&A, Financial Statements, Risk Factors, Quantitative Disclosures,
    Controls and Procedures).

  CUSTOM ANALYTICAL QUERIES:
    Any ad-hoc question against any filing type. The pipeline selects the right
    chunking strategy (semantic for 8-K, template for 10-K/10-Q), enriches with
    macro + market context, and synthesizes a targeted answer.

Every deep analysis is enriched with:
- Macro regime context: FRED yield curve slope, VIX level, credit spreads,
  Fed Funds rate — with derived regime signals (inverted curve = recession risk,
  VIX > 30 = high volatility regime, etc.)
- Market context: Polygon price action (20-day change, avg volume, range),
  market cap, sector, employee count
- XBRL structured financials: Income statement, balance sheet, cash flow
  statement as structured JSON (when available for 10-K/10-Q)

═══ ANALYTICAL GUIDELINES ═══
- Ground EVERY claim in data. Cite FRED series IDs, filing dates, section
  references (e.g., "per Item 7 MD&A of the FY2025 10-K"), and price levels.
- When providing a signal (bullish/bearish/neutral), always state your conviction
  level (high/medium/low) and the key evidence driving it.
- Distinguish between data-backed conclusions and inference. Flag uncertainty.
- For earnings analysis, always note: revenue/EPS vs consensus, guidance changes,
  and management tone shifts relative to prior quarters.
- For filing analysis, prioritize: MD&A narrative, risk factor changes, and
  financial statement trends over boilerplate sections.
- Use precise financial terminology. Format numbers with appropriate units.
- If data is unavailable or an API call fails, say so and work with what you have.
- When a user asks a broad question like "tell me about AAPL," start with
  the most recent filing analysis and supplement with macro + market context.
  Don't just summarize — synthesize an investment view.

═══ RESPONSE FORMAT ═══
For deep analysis responses, structure your output as:

1. SIGNAL & CONVICTION — Lead with the bottom line
2. KEY FINDINGS — The 3-5 most important takeaways, each grounded in data
3. MACRO CONTEXT — How the current regime affects this specific company
4. RISK FLAGS — What could go wrong, with specific triggers
5. CATALYSTS — Near-term events that could move the stock
6. DETAILED ANALYSIS — Full walkthrough of the filing/transcript

For quick data lookups (price, macro indicator, simple filing question), be concise.
Match response depth to query complexity.

You may receive context blocks prefixed with ═══ headers — these contain live data
pulled from APIs and your knowledge base. Use them to ground your response.
"""


class FinanceAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.fred = FREDClient()
        self.edgar = EDGARClient()
        self.polygon = PolygonClient()
        self.vector_store = VectorStore()
        self.conversation: list[dict] = []

    # ── Intent analysis ──────────────────────────────────
    def _detect_intents(self, message: str) -> dict:
        msg_lower = message.lower()
        intents = {
            "macro": any(kw in msg_lower for kw in MACRO_KEYWORDS),
            "filing": any(kw in msg_lower for kw in FILING_KEYWORDS),
            "price": any(kw in msg_lower for kw in PRICE_KEYWORDS),
            "news": any(kw in msg_lower for kw in NEWS_KEYWORDS),
            "rag": True,  # always check knowledge base
            # Market Bridge deep analysis intents
            "deep_analysis": False,
            "earnings_analysis": False,
            "annual_analysis": False,
            "quarterly_analysis": False,
            "custom_analysis": False,
        }
        # Extract tickers
        potential_tickers = TICKER_RE.findall(message)
        # Filter out common English words
        stopwords = {
            "I", "A", "THE", "AND", "OR", "FOR", "IN", "ON", "AT", "TO",
            "IS", "IT", "OF", "BY", "AS", "DO", "IF", "MY", "AN", "BE",
            "NO", "SO", "UP", "HE", "WE", "US", "AM", "HAS", "HAD", "GDP",
            "CPI", "FED", "SEC", "PCE", "VIX", "ISM", "NFP", "FRED",
            "NOT", "BUT", "CAN", "ALL", "NEW", "HOW", "WHO", "WHAT",
            "WHY", "ARE", "WAS", "HIS", "HER", "HIM", "ITS", "MAY",
        }
        tickers = [t for t in potential_tickers if t not in stopwords and len(t) >= 2]
        intents["tickers"] = tickers

        # Deep analysis routing
        has_ticker = bool(tickers)

        if has_ticker and any(kw in msg_lower for kw in DEEP_ANALYSIS_KEYWORDS):
            intents["deep_analysis"] = True

        if has_ticker and any(kw in msg_lower for kw in EARNINGS_KEYWORDS):
            intents["earnings_analysis"] = True
            intents["deep_analysis"] = True

        if has_ticker and any(kw in msg_lower for kw in ANNUAL_REPORT_KEYWORDS):
            intents["annual_analysis"] = True
            intents["deep_analysis"] = True

        if has_ticker and any(kw in msg_lower for kw in QUARTERLY_REPORT_KEYWORDS):
            intents["quarterly_analysis"] = True
            intents["deep_analysis"] = True

        # If user asks to "analyze" a ticker without specifying type,
        # default to custom analysis (ad-hoc query against latest 10-K)
        if intents["deep_analysis"] and not any([
            intents["earnings_analysis"],
            intents["annual_analysis"],
            intents["quarterly_analysis"],
        ]):
            intents["custom_analysis"] = True

        return intents

    # ── Context assembly ─────────────────────────────────
    def _build_context(self, message: str, intents: dict) -> str:
        """Assemble data context blocks based on detected intents."""
        context_parts = []

        # RAG retrieval (always attempt)
        rag_ctx = self.vector_store.format_context(message)
        if rag_ctx:
            context_parts.append(rag_ctx)

        # Macro data
        if intents["macro"]:
            try:
                # Pull specific series if mentioned, else snapshot
                pulled = False
                for key, (sid, label) in MACRO_SERIES.items():
                    if key.replace("_", " ") in message.lower() or sid.lower() in message.lower():
                        val = self.fred.latest_value(sid)
                        if val is not None:
                            context_parts.append(f"═══ {label} ({sid}) ═══\n  Latest: {val:.2f}")
                            pulled = True
                if not pulled:
                    context_parts.append(self.fred.format_snapshot_text())
            except Exception as e:
                logger.error(f"FRED context error: {e}")
                context_parts.append("[FRED data unavailable]")

        # Price data
        tickers = intents.get("tickers", [])
        if intents["price"] and tickers:
            for ticker in tickers[:3]:
                try:
                    context_parts.append(self.polygon.format_quote_text(ticker))
                except Exception as e:
                    logger.error(f"Polygon quote error for {ticker}: {e}")

        # News
        if intents["news"]:
            try:
                ticker = tickers[0] if tickers else None
                context_parts.append(self.polygon.format_news_text(ticker))
            except Exception as e:
                logger.error(f"Polygon news error: {e}")

        # Filing references (standard EDGAR — metadata/summaries only)
        if intents["filing"] and tickers and not intents.get("deep_analysis"):
            for ticker in tickers[:2]:
                try:
                    form = "10-Q" if "10-q" in message.lower() else "10-K"
                    context_parts.append(self.edgar.format_filings_text(ticker, form))
                except Exception as e:
                    logger.error(f"EDGAR context error for {ticker}: {e}")

        # Market Bridge deep analysis
        if intents.get("deep_analysis") and tickers:
            ticker = tickers[0]  # analyze primary ticker
            try:
                from market_bridge.core.pipeline import MarketBridgePipeline
                pipeline = MarketBridgePipeline()

                if intents.get("earnings_analysis"):
                    result = pipeline.analyze_earnings(ticker)
                    context_parts.append(
                        self._format_pipeline_result(result, "EARNINGS DEEP ANALYSIS")
                    )
                elif intents.get("annual_analysis"):
                    result = pipeline.analyze_annual(ticker)
                    context_parts.append(
                        self._format_pipeline_result(result, "ANNUAL REPORT DEEP ANALYSIS")
                    )
                elif intents.get("quarterly_analysis"):
                    result = pipeline.analyze_quarterly(ticker)
                    context_parts.append(
                        self._format_pipeline_result(result, "QUARTERLY REPORT DEEP ANALYSIS")
                    )
                elif intents.get("custom_analysis"):
                    result = pipeline.custom_query(ticker, message)
                    context_parts.append(
                        self._format_pipeline_result(result, "CUSTOM DEEP ANALYSIS")
                    )
            except Exception as e:
                logger.warning(f"Market Bridge analysis failed for {ticker}: {e}")
                context_parts.append(
                    f"═══ MARKET BRIDGE ERROR ═══\n"
                    f"Deep analysis unavailable: {str(e)}\n"
                    f"Falling back to standard context."
                )

        return "\n\n".join(context_parts)

    def _format_pipeline_result(self, result, header: str) -> str:
        """Format a Market Bridge PipelineResult into a context block for Ada."""
        a = result.analysis
        f = result.filing
        sd = a.structured_data

        lines = [
            f"═══ {header}: {result.ticker} ═══",
            f"Signal: {a.signal.upper()} | Conviction: {a.conviction.upper()}",
            f"Filing Date: {f.filed_at if f else 'N/A'}",
            f"Period: {f.period_of_report if f else 'N/A'}",
            "",
            "SUMMARY:",
            a.summary,
            "",
        ]

        if sd and not sd.get("parse_error"):
            lines.append("STRUCTURED FINDINGS:")

            if a.analysis_type == "earnings":
                if sd.get("revenue_surprise"):
                    lines.append(f"  Revenue Surprise: {sd['revenue_surprise']}")
                if sd.get("eps_surprise"):
                    lines.append(f"  EPS Surprise: {sd['eps_surprise']}")
                guidance = sd.get("guidance", {})
                if guidance:
                    lines.append(f"  Guidance: {guidance.get('direction', 'N/A')} — {guidance.get('details', '')}")
                if sd.get("management_tone"):
                    lines.append(f"  Management Tone: {sd['management_tone']}")
                if sd.get("key_themes"):
                    lines.append(f"  Key Themes: {', '.join(sd['key_themes'])}")
                if sd.get("catalyst_triggers"):
                    lines.append(f"  Catalysts: {', '.join(sd['catalyst_triggers'])}")
                if sd.get("risk_flags"):
                    lines.append(f"  Risk Flags: {', '.join(sd['risk_flags'])}")

            elif a.analysis_type in ("annual", "quarterly"):
                fh = sd.get("financial_health", {})
                if fh:
                    for k, v in fh.items():
                        lines.append(f"  {k.replace('_', ' ').title()}: {v}")
                ra = sd.get("risk_assessment", {})
                if ra.get("key_risks"):
                    lines.append(f"  Key Risks: {', '.join(ra['key_risks'][:5])}")
                ss = sd.get("strategic_signals", {})
                if ss.get("growth_initiatives"):
                    lines.append(f"  Growth Initiatives: {', '.join(ss['growth_initiatives'][:3])}")
                mda = sd.get("mda_insights", {})
                if mda.get("management_focus_areas"):
                    lines.append(f"  Mgmt Focus: {', '.join(mda['management_focus_areas'][:3])}")

            elif a.analysis_type == "custom" and sd.get("answer"):
                lines.append(f"  Answer: {sd['answer']}")
                if sd.get("key_data_points"):
                    lines.append(f"  Data Points: {', '.join(sd['key_data_points'][:5])}")

        if result.macro_context:
            lines.append("")
            lines.append(result.macro_context)
        if result.market_context:
            lines.append("")
            lines.append(result.market_context)

        lines.append(f"\n[Chunks analyzed: {a.chunks_used} | Model: {a.model}]")
        return "\n".join(lines)

    # ── Chat (streaming) ─────────────────────────────────
    def chat_stream(self, message: str) -> Generator[str, None, None]:
        """Stream a response, enriching with live data context."""
        intents = self._detect_intents(message)
        context = self._build_context(message, intents)

        # Build user message with injected context
        user_content = message
        if context:
            user_content = f"{message}\n\n--- LIVE DATA CONTEXT ---\n{context}"

        self.conversation.append({"role": "user", "content": user_content})

        # Keep conversation manageable (last 20 turns)
        messages = self.conversation[-20:]

        try:
            with self.client.messages.stream(
                model=Config.LLM_MODEL,
                max_tokens=Config.LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield text

                self.conversation.append({"role": "assistant", "content": full_response})

        except anthropic.APIError as e:
            error_msg = f"⚠️ API Error: {e.message}"
            yield error_msg
            self.conversation.append({"role": "assistant", "content": error_msg})

    def chat(self, message: str) -> str:
        """Non-streaming chat (collects full response)."""
        return "".join(self.chat_stream(message))

    # ── Document ingestion ───────────────────────────────
    def ingest_filing(self, ticker: str, form_type: str = "10-K") -> str:
        """Fetch and ingest a company's latest filing into the vector store."""
        filings = self.edgar.get_recent_filings(ticker, form_type, count=1)
        if not filings:
            return f"No {form_type} filings found for {ticker}."

        filing = filings[0]
        text = self.edgar.fetch_filing_text(filing["url"])
        if not text:
            return f"Could not retrieve filing text for {ticker}."

        n = self.vector_store.ingest_filing(ticker, form_type, text, filing["date"])
        return f"Ingested {ticker} {form_type} ({filing['date']}) — {n} chunks stored."

    def ingest_text(self, text: str, source: str = "user") -> str:
        """Ingest arbitrary text into the knowledge base."""
        n = self.vector_store.ingest_text(text, metadata={"source": source})
        return f"Ingested {n} chunks from {source}."

    # ── Reset ────────────────────────────────────────────
    def clear_history(self):
        self.conversation = []

    def knowledge_base_stats(self) -> str:
        count = self.vector_store.count()
        return f"Knowledge base: {count} document chunks stored."
