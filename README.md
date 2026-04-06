# ◆ Finance Chatbot

Ada Lovelace is a RAG-powered AI finance chatbot that connects Claude's reasoning to live financial data pipelines. It ingests SEC filings (EDGAR 10-K/10-Q), macro indicators (FRED), and market data (Polygon.io) into a ChromaDB vector store, then uses intent detection to dynamically assemble grounded context for each query — so responses cite real numbers, not hallucinated ones. Built on Gradio with streaming chat, a document ingestion panel, and a macro dashboard, Ada is designed as the conversational interface layer for your broader platform architecture, with a native hook point for Access Alpha regime labels and Market Bridge signal routing. Think of it as your research desk analyst that actually checks the tape before it talks.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Gradio UI                          │
│  Chat │ Knowledge Base │ Macro Dashboard │ Market    │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Finance Agent  │  ← intent detection
              │  (Orchestrator) │  ← context assembly
              └──┬───┬───┬───┬─┘
                 │   │   │   │
    ┌────────┐ ┌┴┐ ┌┴┐ ┌┴┐ ┌┴────────┐
    │ChromaDB│ │F│ │E│ │P│ │ Claude   │
    │  RAG   │ │R│ │D│ │o│ │   API    │
    │        │ │E│ │G│ │l│ │(Sonnet)  │
    └────────┘ │D│ │A│ │y│ └──────────┘
               │ │ │R│ │g│
               └─┘ └─┘ │o│
                        │n│
                        └─┘
```

## Stack

| Layer           | Technology                          |
|-----------------|-------------------------------------|
| LLM             | Claude API (Sonnet 4)               |
| RAG             | ChromaDB (local, cosine similarity) |
| Macro Data      | FRED (Federal Reserve)              |
| SEC Filings     | EDGAR (10-K, 10-Q)                  |
| Market Data     | Polygon.io (quotes, news)           |
| Frontend        | Gradio                              |

## Quick Start

```bash
# 1. Clone and enter
cd finance-chatbot

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Launch
python app.py
```

## API Keys

| Service   | Free Tier | URL                                       |
|-----------|-----------|-------------------------------------------|
| Anthropic | No        | https://console.anthropic.com             |
| FRED      | Yes       | https://fred.stlouisfed.org/docs/api      |
| Polygon   | Yes       | https://polygon.io                        |

## Features

### Chat
- Streaming responses from Claude with auto-injected live data context
- Intent detection routes queries to appropriate data sources
- Full conversation history with context window management

### Knowledge Base (RAG)
- Ingest SEC filings directly by ticker + form type
- Paste research notes, earnings transcripts, etc.
- Upload text files (.txt, .md, .csv)
- ChromaDB cosine similarity retrieval

### Macro Dashboard
- One-click FRED macro snapshot (18 indicators)
- Fed Funds, GDP, CPI, unemployment, yield curve, VIX, and more

### Market Data
- Polygon.io quotes with OHLCV
- Filtered or general market news

## Integration with Access Alpha / Market Bridge

The agent's intent detection and data pipeline architecture is designed to accept
regime labels from Access Alpha as context enrichment:

```python
# In core/__init__.py, extend _build_context():
if regime_label:
    context_parts.append(f"═══ REGIME: {regime_label} ═══")
```

## Project Structure

```
finance-chatbot/
├── app.py                  # Gradio UI
├── config.py               # Environment config
├── core/
│   └── __init__.py         # FinanceAgent orchestrator
├── data_sources/
│   ├── __init__.py         # FRED client
│   ├── edgar_client.py     # SEC EDGAR client
│   └── polygon_client.py   # Polygon.io client
├── rag/
│   └── __init__.py         # ChromaDB vector store
├── requirements.txt
├── .env.example
└── README.md
```
