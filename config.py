"""
Finance Chatbot Configuration
──────────────────────────────
All API keys loaded from environment or .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ──────────────────────────────────────────────
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # ── Finance Data APIs ────────────────────────────────
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

    # ── RAG / Vector Store ───────────────────────────────
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "finance_docs")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

    # ── EDGAR ────────────────────────────────────────────
    EDGAR_USER_AGENT = os.getenv(
        "EDGAR_USER_AGENT", "FinanceChatbot research@example.com"
    )
    EDGAR_DOWNLOAD_DIR = os.getenv("EDGAR_DOWNLOAD_DIR", "./edgar_filings")

    # ── Market Bridge (sec-api.io) ───────────────────────
    SEC_API_KEY = os.getenv("SEC_API_KEY", "")

    # ── Gradio ───────────────────────────────────────────
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
