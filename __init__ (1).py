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


SYSTEM_PROMPT = """\
You are a senior quantitative finance analyst and research assistant.
You have access to live macro data (FRED), SEC filings (EDGAR), market data (Polygon.io),
and a vector knowledge base of ingested financial documents.

Guidelines:
- Ground every claim in data. Cite sources (FRED series IDs, filing dates, etc.)
- When discussing macro conditions, reference specific indicators and their recent trends
- For company analysis, reference filing excerpts and price action
- Flag uncertainty clearly — distinguish data-backed conclusions from inference
- Use precise financial terminology
- Format numbers with appropriate precision and units
- If data is unavailable, say so rather than guessing

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

        # Filing references
        if intents["filing"] and tickers:
            for ticker in tickers[:2]:
                try:
                    form = "10-Q" if "10-q" in message.lower() else "10-K"
                    context_parts.append(self.edgar.format_filings_text(ticker, form))
                except Exception as e:
                    logger.error(f"EDGAR context error for {ticker}: {e}")

        return "\n\n".join(context_parts)

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
