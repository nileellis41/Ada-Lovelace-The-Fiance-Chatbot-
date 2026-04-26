"""
Finance Chatbot — Gradio Application
──────────────────────────────────────
RAG-powered financial research assistant with live data integration.

Launch:  python app.py
"""
import logging
import os

import gradio as gr

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


# ═════════════════════════════════════════════════════════
#  Lazy-init agent (so the UI loads even without API keys)
# ═════════════════════════════════════════════════════════
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from core import FinanceAgent
        _agent = FinanceAgent()
    return _agent


# ═════════════════════════════════════════════════════════
#  Chat handler (streaming)
# ═════════════════════════════════════════════════════════
def chat_respond(message: str, history: list[dict]):
    """Gradio chatbot handler with streaming."""
    if not message.strip():
        return "", history

    agent = get_agent()
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for token in agent.chat_stream(message):
        history[-1]["content"] += token
        yield "", history


def clear_chat():
    get_agent().clear_history()
    return [], ""


# ═════════════════════════════════════════════════════════
#  Document ingestion handlers
# ═════════════════════════════════════════════════════════
def ingest_filing_handler(ticker: str, form_type: str):
    if not ticker.strip():
        return "⚠️ Enter a ticker symbol."
    agent = get_agent()
    result = agent.ingest_filing(ticker.strip().upper(), form_type)
    stats = agent.knowledge_base_stats()
    return f"{result}\n{stats}"


def ingest_text_handler(text: str, source_label: str):
    if not text.strip():
        return "⚠️ Paste some text to ingest."
    agent = get_agent()
    result = agent.ingest_text(text.strip(), source_label or "manual")
    stats = agent.knowledge_base_stats()
    return f"{result}\n{stats}"


def ingest_file_handler(file):
    if file is None:
        return "⚠️ Upload a file."
    try:
        agent = get_agent()
        filename = os.path.basename(file.name)
        if filename.lower().endswith(".pdf"):
            result = agent.ingest_pdf(file.name)
        else:
            with open(file.name, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            result = agent.ingest_text(text, source=filename)
        stats = agent.knowledge_base_stats()
        return f"{result}\n{stats}"
    except Exception as e:
        return f"⚠️ Error reading file: {e}"


def get_kb_stats():
    return get_agent().knowledge_base_stats()


# ═════════════════════════════════════════════════════════
#  Macro snapshot handler
# ═════════════════════════════════════════════════════════
def fetch_macro_snapshot():
    try:
        from data_sources import FREDClient
        fred = FREDClient()
        snap = fred.macro_snapshot()
        rows = []
        for _, info in snap.items():
            val = info["value"]
            rows.append([info["label"], info["series_id"], f"{val:.2f}" if val is not None else "N/A"])
        return rows
    except Exception as e:
        return [[f"Error: {e}", "", ""]]


# ═════════════════════════════════════════════════════════
#  Market quote handler
# ═════════════════════════════════════════════════════════
def fetch_quote(ticker: str):
    if not ticker.strip():
        return "Enter a ticker."
    try:
        from data_sources.polygon_client import PolygonClient
        poly = PolygonClient()
        return poly.format_quote_text(ticker.strip())
    except Exception as e:
        return f"Error: {e}"


def fetch_news(ticker: str):
    try:
        from data_sources.polygon_client import PolygonClient
        poly = PolygonClient()
        t = ticker.strip() if ticker.strip() else None
        return poly.format_news_text(t)
    except Exception as e:
        return f"Error: {e}"


# ═════════════════════════════════════════════════════════
#  Deep Analysis handlers (Market Bridge pipeline)
# ═════════════════════════════════════════════════════════
# Store the last result so the inspector can show chunks + JSON
_last_pipeline_result = None


def run_deep_analysis(ticker: str, analysis_type: str, custom_question: str):
    global _last_pipeline_result
    ticker = ticker.strip().upper()
    if not ticker:
        return "⚠️ Enter a ticker symbol.", "", ""
    try:
        from market_bridge.core.pipeline import MarketBridgePipeline
        pipeline = MarketBridgePipeline()

        if analysis_type == "Earnings (8-K)":
            result = pipeline.analyze_earnings(ticker)
        elif analysis_type == "Annual (10-K)":
            result = pipeline.analyze_annual(ticker)
        elif analysis_type == "Quarterly (10-Q)":
            result = pipeline.analyze_quarterly(ticker)
        else:  # Custom Query
            if not custom_question.strip():
                return "⚠️ Enter a question for custom analysis.", "", ""
            filing_type = "10-K"
            if "10-q" in custom_question.lower() or "quarterly" in custom_question.lower():
                filing_type = "10-Q"
            elif "8-k" in custom_question.lower() or "earnings" in custom_question.lower():
                filing_type = "8-K"
            result = pipeline.custom_query(ticker, custom_question, filing_type)

        _last_pipeline_result = result

        a = result.analysis
        f = result.filing
        summary_md = f"""## {ticker} — {analysis_type}

**Signal:** `{a.signal.upper()}` | **Conviction:** `{a.conviction.upper()}`
**Filing Date:** {f.filed_at if f else 'N/A'} | **Period:** {f.period_of_report if f else 'N/A'}
**Chunks analyzed:** {a.chunks_used} | **Model:** {a.model}

---

{a.summary}
"""
        import json
        raw_json = json.dumps(result.to_dict(), indent=2)

        chunks_lines = []
        for i, chunk in enumerate(result.chunks, 1):
            chunks_lines.append(
                f"**Chunk {i}** — type: `{chunk.chunk_type}` | section: `{chunk.section_label}` | {len(chunk.text)} chars\n"
                f"> {chunk.text[:300].strip()}{'...' if len(chunk.text) > 300 else ''}\n"
            )
        chunks_md = "\n".join(chunks_lines) if chunks_lines else "_No chunks available._"

        return summary_md, chunks_md, raw_json

    except Exception as e:
        logger.error(f"Deep analysis error: {e}", exc_info=True)
        return f"⚠️ Error: {e}", "", ""


# ═════════════════════════════════════════════════════════
#  Custom CSS (dark theme, clean finance aesthetic)
# ═════════════════════════════════════════════════════════
# (CSS is inline in the HTML block below)


# ═════════════════════════════════════════════════════════
#  Build the Gradio UI
# ═════════════════════════════════════════════════════════
def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Finance Chatbot",
    ) as app:

        # ── Header ───────────────────────────────────────
        gr.HTML("""
        <style>
        .gradio-container {
            max-width: 1200px !important;
        }
        .header-banner {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 16px;
            text-align: center;
        }
        .header-banner h1 {
            background: linear-gradient(90deg, #00d2ff, #7b68ee, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .header-banner p {
            color: #8888aa;
            margin: 8px 0 0 0;
            font-size: 0.9rem;
        }
        </style>
        <div class="header-banner">
            <h1>◆ FINANCE CHATBOT</h1>
            <p>RAG-powered research assistant · Claude LLM · FRED · EDGAR · Polygon.io</p>
        </div>
        """)

        with gr.Tabs():
            # ━━━━━ TAB 1: CHAT ━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("💬 Chat", id="chat"):
                chatbot = gr.Chatbot(
                    height=520,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. What's the current Fed Funds rate and how does it compare to 2023?",
                        show_label=False,
                        scale=6,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)

                # Example prompts
                gr.Examples(
                    examples=[
                        "Give me a macro snapshot — what do key indicators say about recession risk?",
                        "Pull the latest AAPL 10-K highlights and summarize risk factors.",
                        "What's the yield curve telling us right now? Reference T10Y2Y and T10Y3M.",
                        "Compare NVDA and AMD — price action and recent filing sentiment.",
                        "What's in my knowledge base right now?",
                    ],
                    inputs=msg_input,
                    label="Quick Prompts",
                )

                # Wire up chat events
                send_btn.click(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
                msg_input.submit(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
                clear_btn.click(clear_chat, outputs=[chatbot, msg_input])

            # ━━━━━ TAB 2: KNOWLEDGE BASE ━━━━━━━━━━━━━━━━━
            with gr.Tab("📚 Knowledge Base", id="kb"):
                gr.Markdown("### Ingest Documents into the Vector Store")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**SEC Filing Ingestion**")
                        filing_ticker = gr.Textbox(label="Ticker", placeholder="AAPL")
                        filing_type = gr.Dropdown(
                            choices=["10-K", "10-Q"],
                            value="10-K",
                            label="Form Type",
                        )
                        filing_btn = gr.Button("Fetch & Ingest Filing", variant="primary")
                        filing_output = gr.Textbox(label="Result", lines=3, interactive=False)

                    with gr.Column():
                        gr.Markdown("**Manual Text Ingestion**")
                        manual_text = gr.Textbox(
                            label="Paste Text",
                            placeholder="Paste research notes, earnings call transcripts, etc.",
                            lines=6,
                        )
                        source_label = gr.Textbox(label="Source Label", placeholder="e.g. Q3 earnings call")
                        text_btn = gr.Button("Ingest Text", variant="primary")
                        text_output = gr.Textbox(label="Result", lines=3, interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**File Upload**")
                        file_upload = gr.File(label="Upload .txt / .md / .csv / .pdf", file_types=[".txt", ".md", ".csv", ".pdf"])
                        file_btn = gr.Button("Ingest File", variant="primary")
                        file_output = gr.Textbox(label="Result", lines=3, interactive=False)

                    with gr.Column():
                        gr.Markdown("**Knowledge Base Status**")
                        kb_stats_output = gr.Textbox(label="Stats", lines=3, interactive=False)
                        kb_refresh_btn = gr.Button("Refresh Stats")

                filing_btn.click(ingest_filing_handler, [filing_ticker, filing_type], filing_output)
                text_btn.click(ingest_text_handler, [manual_text, source_label], text_output)
                file_btn.click(ingest_file_handler, [file_upload], file_output)
                kb_refresh_btn.click(get_kb_stats, outputs=kb_stats_output)

            # ━━━━━ TAB 3: MACRO DASHBOARD ━━━━━━━━━━━━━━━
            with gr.Tab("📊 Macro Dashboard", id="macro"):
                gr.Markdown("### Live FRED Macro Indicators")
                macro_btn = gr.Button("Fetch Macro Snapshot", variant="primary")
                macro_table = gr.Dataframe(
                    headers=["Indicator", "Series ID", "Latest Value"],
                    interactive=False,
                )
                macro_btn.click(fetch_macro_snapshot, outputs=macro_table)

            # ━━━━━ TAB 4: MARKET DATA ━━━━━━━━━━━━━━━━━━━
            with gr.Tab("📈 Market Data", id="market"):
                gr.Markdown("### Polygon.io Market Data")
                with gr.Row():
                    with gr.Column():
                        quote_ticker = gr.Textbox(label="Ticker", placeholder="AAPL")
                        quote_btn = gr.Button("Get Quote", variant="primary")
                        quote_output = gr.Textbox(label="Quote", lines=6, interactive=False)

                    with gr.Column():
                        news_ticker = gr.Textbox(label="Ticker (optional)", placeholder="Leave blank for general news")
                        news_btn = gr.Button("Get News", variant="primary")
                        news_output = gr.Textbox(label="News", lines=8, interactive=False)

                quote_btn.click(fetch_quote, [quote_ticker], quote_output)
                news_btn.click(fetch_news, [news_ticker], news_output)

            # ━━━━━ TAB 5: DEEP ANALYSIS ━━━━━━━━━━━━━━━━━
            with gr.Tab("🔬 Deep Analysis", id="deep_analysis"):
                gr.Markdown("### Market Bridge — Structured Filing Analysis")
                gr.Markdown(
                    "Run the full Market Bridge pipeline: SEC filing → chunking → Claude synthesis → structured JSON signal."
                )

                with gr.Row():
                    da_ticker = gr.Textbox(label="Ticker", placeholder="e.g. AAPL", scale=1)
                    da_type = gr.Radio(
                        choices=["Earnings (8-K)", "Annual (10-K)", "Quarterly (10-Q)", "Custom Query"],
                        value="Earnings (8-K)",
                        label="Analysis Type",
                        scale=3,
                    )

                da_question = gr.Textbox(
                    label="Custom Question (required for Custom Query)",
                    placeholder="e.g. What are the main AI revenue drivers?",
                    visible=False,
                )

                da_type.change(
                    fn=lambda t: gr.update(visible=(t == "Custom Query")),
                    inputs=da_type,
                    outputs=da_question,
                )

                da_run_btn = gr.Button("Run Analysis", variant="primary")

                with gr.Tabs():
                    with gr.Tab("Analysis"):
                        da_summary = gr.Markdown(label="Result")
                    with gr.Tab("Chunks"):
                        da_chunks = gr.Markdown(label="Chunk Inspector")
                    with gr.Tab("Raw JSON"):
                        da_json = gr.Code(language="json", label="Structured Output")

                da_run_btn.click(
                    run_deep_analysis,
                    inputs=[da_ticker, da_type, da_question],
                    outputs=[da_summary, da_chunks, da_json],
                )

            # ━━━━━ TAB 6: CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("⚙️ Config", id="config"):
                gr.Markdown("### Configuration Status")
                api_status = []
                for name, key in [
                    ("Anthropic API", Config.ANTHROPIC_API_KEY),
                    ("FRED API", Config.FRED_API_KEY),
                    ("Polygon API", Config.POLYGON_API_KEY),
                ]:
                    status = "✅ Set" if key else "❌ Not set"
                    api_status.append(f"**{name}**: {status}")

                gr.Markdown("\n\n".join(api_status))

                gr.Markdown(f"""
**Model**: `{Config.LLM_MODEL}`
**Max Tokens**: `{Config.LLM_MAX_TOKENS}`
**ChromaDB Dir**: `{Config.CHROMA_PERSIST_DIR}`
**Chunk Size**: `{Config.CHUNK_SIZE}` / Overlap: `{Config.CHUNK_OVERLAP}`
**Top-K Results**: `{Config.TOP_K_RESULTS}`
                """)

                gr.Markdown("""
### Setup Instructions

1. Copy `.env.example` to `.env` and add your API keys:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   FRED_API_KEY=...           # Free at https://fred.stlouisfed.org/docs/api/api_key.html
   POLYGON_API_KEY=...        # Free tier at https://polygon.io
   ```
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
                """)

    return app


# ═════════════════════════════════════════════════════════
#  Launch
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=Config.GRADIO_SERVER_NAME,
        server_port=Config.GRADIO_SERVER_PORT,
        share=Config.GRADIO_SHARE,
        show_error=True,
    )
