"""
Ada Lovelace — Web Application + OpenAI-compatible API Server
──────────────────────────────────────────────────────────────
Serves the web UI at / and all API endpoints.

Launch:  python server.py
         → http://127.0.0.1:8000
"""
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Ada Lovelace Finance API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy agent init ──────────────────────────────────────
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from core import FinanceAgent
        _agent = FinanceAgent()
        logger.info("FinanceAgent initialized")
    return _agent


# ── Pydantic models ──────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "ada-lovelace"
    messages: List[Message]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatMessageRequest(BaseModel):
    message: str


class IngestFilingRequest(BaseModel):
    ticker: str
    form_type: str = "10-K"


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"


class AnalysisRequest(BaseModel):
    ticker: str
    analysis_type: str
    custom_question: str = ""


# ── Web UI ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = BASE_DIR / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


# ── Chat API ─────────────────────────────────────────────
@app.post("/api/chat")
async def api_chat(request: ChatMessageRequest):
    agent = get_agent()

    async def event_stream():
        try:
            for token in agent.chat_stream(request.message):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/clear")
def api_chat_clear():
    get_agent().clear_history()
    return {"status": "ok"}


# ── Knowledge Base API ───────────────────────────────────
@app.post("/api/ingest/filing")
def api_ingest_filing(request: IngestFilingRequest):
    agent = get_agent()
    result = agent.ingest_filing(request.ticker.upper(), request.form_type)
    return {"result": result, "stats": agent.knowledge_base_stats()}


@app.post("/api/ingest/text")
def api_ingest_text(request: IngestTextRequest):
    if not request.text.strip():
        raise HTTPException(400, "No text provided")
    agent = get_agent()
    result = agent.ingest_text(request.text.strip(), request.source or "manual")
    return {"result": result, "stats": agent.knowledge_base_stats()}


@app.post("/api/ingest/file")
async def api_ingest_file(file: UploadFile = File(...)):
    agent = get_agent()
    content = await file.read()
    filename = file.filename or "upload"

    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if filename.lower().endswith(".pdf"):
            result = agent.ingest_pdf(tmp_path)
        else:
            text = content.decode("utf-8", errors="replace")
            result = agent.ingest_text(text, source=filename)
        return {"result": result, "stats": agent.knowledge_base_stats()}
    finally:
        os.unlink(tmp_path)


@app.get("/api/kb/stats")
def api_kb_stats():
    return {"stats": get_agent().knowledge_base_stats()}


# ── Market Data API ──────────────────────────────────────
@app.get("/api/macro")
def api_macro():
    try:
        from data_sources import FREDClient
        fred = FREDClient()
        snap = fred.macro_snapshot()
        rows = []
        for _, info in snap.items():
            val = info["value"]
            rows.append({
                "label": info["label"],
                "series_id": info["series_id"],
                "value": f"{val:.2f}" if val is not None else "N/A",
            })
        return {"indicators": rows}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/quote/{ticker}")
def api_quote(ticker: str):
    try:
        from data_sources.polygon_client import PolygonClient
        return {"text": PolygonClient().format_quote_text(ticker.upper())}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/news")
def api_news(ticker: str = Query(default="")):
    try:
        from data_sources.polygon_client import PolygonClient
        t = ticker.strip().upper() if ticker.strip() else None
        return {"text": PolygonClient().format_news_text(t)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Deep Analysis API ────────────────────────────────────
@app.post("/api/analysis")
def api_analysis(request: AnalysisRequest):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "Ticker required")

    try:
        from market_bridge.core.pipeline import MarketBridgePipeline
        pipeline = MarketBridgePipeline()

        atype = request.analysis_type
        if atype == "earnings":
            result = pipeline.analyze_earnings(ticker)
        elif atype == "annual":
            result = pipeline.analyze_annual(ticker)
        elif atype == "quarterly":
            result = pipeline.analyze_quarterly(ticker)
        else:
            if not request.custom_question.strip():
                raise HTTPException(400, "Custom question required")
            result = pipeline.custom_query(ticker, request.custom_question)

        a = result.analysis
        f = result.filing

        return {
            "ticker": ticker,
            "signal": a.signal,
            "conviction": a.conviction,
            "filing_date": f.filed_at if f else "N/A",
            "period": f.period_of_report if f else "N/A",
            "chunks_used": a.chunks_used,
            "model": a.model,
            "summary": a.summary,
            "chunks": [
                {
                    "type": c.chunk_type,
                    "section": c.section_label,
                    "length": len(c.text),
                    "preview": c.text[:300],
                }
                for c in result.chunks
            ],
            "raw": result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── Config API ───────────────────────────────────────────
@app.get("/api/config")
def api_config():
    return {
        "anthropic": bool(Config.ANTHROPIC_API_KEY),
        "fred": bool(Config.FRED_API_KEY),
        "polygon": bool(Config.POLYGON_API_KEY),
        "sec_api": bool(getattr(Config, "SEC_API_KEY", "")),
        "model": Config.LLM_MODEL,
        "max_tokens": Config.LLM_MAX_TOKENS,
        "chroma_dir": Config.CHROMA_PERSIST_DIR,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "top_k": Config.TOP_K_RESULTS,
    }


# ── OpenAI-compatible API (for external clients like Open WebUI) ─────────
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "ada-lovelace", "object": "model", "created": 0, "owned_by": "ada"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message in request.")

    query = user_messages[-1].content
    agent = get_agent()

    agent.conversation = [
        {"role": m.role, "content": m.content}
        for m in request.messages[:-1]
    ]

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        async def event_stream():
            try:
                for token in agent.chat_stream(query):
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": created, "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                err = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": created, "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {e}]"}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    full = agent.chat(query)
    return {
        "id": req_id, "object": "chat.completion", "created": created,
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Launch ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ◆ Ada Lovelace — Finance Research AI")
    print("  → Open in browser: http://localhost:8000\n")
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
