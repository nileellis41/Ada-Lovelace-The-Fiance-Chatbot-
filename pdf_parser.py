"""
PDF Parser
──────────
Parses research PDFs (sector dashboards, macro notes, earnings decks) into
text + table chunks, and extracts a structured summary for Ada's RAG store.

Strategy (hybrid):
  1. Try pdfplumber text + table extraction (works for native-text PDFs).
  2. If extraction is empty (print-to-PDF, scans, image-only exports),
     fall back to Claude vision by rasterizing pages.
  3. Rule-based summarizers for known formats (e.g. S&P Select Industry
     Dashboard) produce structured signals cheaply; LLM fallback handles
     anything unrecognized.
"""

import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import pdfplumber

from config import Config

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────
MIN_CHARS_PER_PAGE = 50            # below this → treat as image-only
MAX_VISION_PAGES = 8               # cap on pages sent to Claude vision
RASTER_DPI = 150                   # pdftoppm DPI for vision fallback
MAX_CHUNK_CHARS = 1800             # per-chunk size for RAG

# ── Known format signatures (rule-based fast path) ──────
SP_DASHBOARD_MARKERS = (
    "S&P Select Industry",
    "Select Industry Dashboard",
    "QUARTERLY PERFORMANCE SUMMARY",
)


# ═════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════

@dataclass
class PDFChunk:
    """A single ingestable unit from a PDF."""
    text: str
    chunk_type: str              # "text" | "table" | "vision"
    page: int
    section_label: str = ""      # e.g. "Table 1", "Page 3 narrative"

    def as_metadata(self) -> dict:
        return {
            "chunk_type": self.chunk_type,
            "page": self.page,
            "section_label": self.section_label,
        }


@dataclass
class ParsedPDF:
    """Full parse result: chunks for RAG + structured summary for the UI."""
    source: str                             # filename / path
    page_count: int
    extraction_method: str                  # "text" | "vision" | "mixed"
    chunks: list[PDFChunk] = field(default_factory=list)
    structured_summary: dict = field(default_factory=dict)
    format_detected: str = "generic"        # "sp_industry_dashboard" | "generic" | ...
    raw_text: str = ""                      # concatenated text (may be empty)

    def total_chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)


# ═════════════════════════════════════════════════════════
# PDFParser
# ═════════════════════════════════════════════════════════

class PDFParser:
    """Hybrid text+vision PDF parser with rule-based summarizers."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    # ── Public entry point ──────────────────────────────
    def parse(self, path: str) -> ParsedPDF:
        """Parse a PDF end-to-end. Safe to call on any PDF."""
        source = os.path.basename(path)
        logger.info(f"Parsing PDF: {source}")

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            text_chunks, table_chunks, raw_text = self._extract_text_and_tables(pdf)

        # Decide: did text extraction work?
        avg_chars = (len(raw_text) / page_count) if page_count else 0
        needs_vision = avg_chars < MIN_CHARS_PER_PAGE

        vision_chunks: list[PDFChunk] = []
        vision_text = ""
        if needs_vision:
            logger.info(f"Text extraction sparse ({avg_chars:.0f} chars/page) — using Claude vision")
            vision_chunks, vision_text = self._extract_via_vision(path, page_count)
            extraction_method = "vision"
        else:
            extraction_method = "mixed" if table_chunks else "text"

        chunks = text_chunks + table_chunks + vision_chunks
        combined_text = raw_text + "\n\n" + vision_text

        # Format detection + structured summary
        format_detected = self._detect_format(combined_text)
        structured = self._summarize(combined_text, format_detected, source)

        parsed = ParsedPDF(
            source=source,
            page_count=page_count,
            extraction_method=extraction_method,
            chunks=chunks,
            structured_summary=structured,
            format_detected=format_detected,
            raw_text=combined_text,
        )
        logger.info(
            f"Parsed {source}: {page_count} pages, {len(chunks)} chunks, "
            f"{extraction_method} extraction, format={format_detected}"
        )
        return parsed

    # ── Text + table extraction via pdfplumber ──────────
    def _extract_text_and_tables(self, pdf) -> tuple[list[PDFChunk], list[PDFChunk], str]:
        text_chunks: list[PDFChunk] = []
        table_chunks: list[PDFChunk] = []
        all_text_parts: list[str] = []

        for i, page in enumerate(pdf.pages, start=1):
            # Text
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Text extraction failed on page {i}: {e}")
                page_text = ""

            if page_text.strip():
                all_text_parts.append(page_text)
                for j, piece in enumerate(self._split_text(page_text)):
                    text_chunks.append(PDFChunk(
                        text=piece,
                        chunk_type="text",
                        page=i,
                        section_label=f"Page {i} narrative" + (f" [{j+1}]" if j else ""),
                    ))

            # Tables → render as markdown so Claude reads them natively
            try:
                tables = page.extract_tables() or []
            except Exception as e:
                logger.warning(f"Table extraction failed on page {i}: {e}")
                tables = []

            for k, table in enumerate(tables, start=1):
                md = self._table_to_markdown(table)
                if md:
                    table_chunks.append(PDFChunk(
                        text=md,
                        chunk_type="table",
                        page=i,
                        section_label=f"Page {i} Table {k}",
                    ))

        raw_text = "\n\n".join(all_text_parts)
        return text_chunks, table_chunks, raw_text

    @staticmethod
    def _split_text(text: str) -> list[str]:
        """Split a page of text into RAG-sized pieces on paragraph boundaries."""
        text = text.strip()
        if len(text) <= MAX_CHUNK_CHARS:
            return [text]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 > MAX_CHUNK_CHARS and buf:
                chunks.append(buf)
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append(buf)
        return chunks

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Render a pdfplumber table as a markdown table, skipping empties."""
        cleaned = [
            [(cell or "").strip() for cell in row]
            for row in table
            if row and any((cell or "").strip() for cell in row)
        ]
        if len(cleaned) < 2:
            return ""
        header = cleaned[0]
        body = cleaned[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in body:
            # Pad short rows
            row = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(row[:len(header)]) + " |")
        return "\n".join(lines)

    # ── Vision fallback for image-only PDFs ─────────────
    def _extract_via_vision(self, path: str, page_count: int) -> tuple[list[PDFChunk], str]:
        """Rasterize pages and send to Claude vision for text extraction."""
        pages_to_process = min(page_count, MAX_VISION_PAGES)
        chunks: list[PDFChunk] = []
        combined_parts: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            for page_num in range(1, pages_to_process + 1):
                img_path = self._rasterize_page(path, page_num, tmp)
                if not img_path:
                    continue
                extracted = self._vision_read_page(img_path, page_num)
                if not extracted:
                    continue
                combined_parts.append(f"[Page {page_num}]\n{extracted}")
                for j, piece in enumerate(self._split_text(extracted)):
                    chunks.append(PDFChunk(
                        text=piece,
                        chunk_type="vision",
                        page=page_num,
                        section_label=f"Page {page_num} vision" + (f" [{j+1}]" if j else ""),
                    ))

        if page_count > MAX_VISION_PAGES:
            logger.info(f"Vision capped at {MAX_VISION_PAGES} of {page_count} pages")

        return chunks, "\n\n".join(combined_parts)

    @staticmethod
    def _rasterize_page(pdf_path: str, page_num: int, outdir: str) -> Optional[str]:
        """Use pdftoppm to render a single page to JPEG. Returns file path or None."""
        prefix = os.path.join(outdir, f"page_{page_num}")
        try:
            subprocess.run(
                ["pdftoppm", "-jpeg", "-r", str(RASTER_DPI),
                 "-f", str(page_num), "-l", str(page_num), pdf_path, prefix],
                check=True, capture_output=True, timeout=30,
            )
        except FileNotFoundError:
            logger.error("pdftoppm not installed — vision fallback unavailable. "
                         "Install poppler-utils (apt) or poppler (brew).")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"pdftoppm failed on page {page_num}: {e.stderr.decode()[:200]}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"pdftoppm timed out on page {page_num}")
            return None

        # pdftoppm zero-pads filenames; find the actual output
        for fname in os.listdir(outdir):
            if fname.startswith(f"page_{page_num}") and fname.endswith(".jpg"):
                return os.path.join(outdir, fname)
        return None

    def _vision_read_page(self, img_path: str, page_num: int) -> str:
        """Send a rasterized page to Claude for text + table extraction."""
        try:
            with open(img_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode()
        except Exception as e:
            logger.error(f"Failed to read rasterized page {page_num}: {e}")
            return ""

        prompt = (
            "Extract all text, numbers, and tabular data from this research PDF page. "
            "Preserve table structure as markdown tables. Capture every percentage, "
            "metric, ticker, and date. Omit decorative elements. Output only the "
            "extracted content — no preamble."
        )

        try:
            resp = self.client.messages.create(
                model=Config.LLM_MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()
        except anthropic.APIError as e:
            logger.error(f"Vision API error on page {page_num}: {e}")
            return ""

    # ── Format detection ────────────────────────────────
    @staticmethod
    def _detect_format(text: str) -> str:
        if any(marker in text for marker in SP_DASHBOARD_MARKERS):
            return "sp_industry_dashboard"
        return "generic"

    # ── Structured summary (hybrid: rules → LLM) ────────
    def _summarize(self, text: str, format_detected: str, source: str) -> dict:
        """Dispatch to the right summarizer based on detected format."""
        if not text.strip():
            return {"error": "no extractable content"}

        if format_detected == "sp_industry_dashboard":
            rule_based = self._summarize_sp_dashboard(text)
            if rule_based.get("leaders") or rule_based.get("laggards"):
                rule_based["method"] = "rule_based"
                return rule_based
            # Fall through to LLM if rules didn't land

        return self._summarize_llm(text, source)

    # ── Rule-based: S&P Select Industry Dashboard ──────
    @staticmethod
    def _summarize_sp_dashboard(text: str) -> dict:
        """
        Extract sector performance from S&P Select Industry Dashboard text.

        Expected line shape (from pdfplumber or vision):
          "Oil & Gas Expl & Prod  18.79%  44.67%  44.67%  41.72%"
          columns: INDEX, MTD, QTD, 3M, 12M
        """
        # Match: name (words/spaces/&) then 4 signed percentages
        pattern = re.compile(
            r"([A-Za-z&\s/\-]+?)\s+"
            r"(-?\d+\.\d+)%\s+"
            r"(-?\d+\.\d+)%\s+"
            r"(-?\d+\.\d+)%\s+"
            r"(-?\d+\.\d+)%"
        )
        rows = []
        header_tokens = {"INDEX", "METRICS", "MTD", "QTD", "3M", "12M"}
        for m in pattern.finditer(text):
            # Collapse any internal whitespace/newlines in the captured name.
            name = re.sub(r"\s+", " ", m.group(1)).strip()
            # Strip leading header-token fragments or stray single letters
            # bleeding in from the row above (e.g. "M" from "MTD").
            while name:
                first = name.split(" ", 1)[0]
                if first.upper() in header_tokens or len(first) == 1:
                    parts = name.split(" ", 1)
                    name = parts[1] if len(parts) > 1 else ""
                else:
                    break
            if len(name) < 3 or name.upper() in header_tokens:
                continue
            rows.append({
                "sector": name,
                "mtd_pct": float(m.group(2)),
                "qtd_pct": float(m.group(3)),
                "three_month_pct": float(m.group(4)),
                "twelve_month_pct": float(m.group(5)),
            })

        if not rows:
            return {}

        # Leaders / laggards by QTD (the dashboard's primary ranking column)
        by_qtd = sorted(rows, key=lambda r: r["qtd_pct"], reverse=True)
        leaders = by_qtd[:3]
        laggards = by_qtd[-3:][::-1]

        # As-of date
        date_match = re.search(
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}",
            text,
        )
        as_of = date_match.group(0) if date_match else None

        return {
            "format": "S&P Select Industry Dashboard",
            "as_of": as_of,
            "sector_count": len(rows),
            "leaders": leaders,
            "laggards": laggards,
            "all_sectors": rows,
        }

    # ── LLM-based fallback summarizer ───────────────────
    def _summarize_llm(self, text: str, source: str) -> dict:
        """Ask Claude for a structured summary of an unknown-format research PDF."""
        # Cap input to keep this cheap
        snippet = text[:12000]
        prompt = f"""You are Ada Lovelace's PDF analysis module. Extract a structured
summary of this research document. Return ONLY valid JSON, no preamble or fences.

Required schema:
{{
  "document_type": string,              // e.g. "sector report", "earnings deck", "macro note"
  "as_of": string | null,                // publication/data date if findable
  "headline": string,                    // one-sentence thesis
  "key_metrics": [string],               // 3-7 most important numbers with labels
  "signals": [                           // bullish/bearish/neutral calls
    {{"topic": string, "direction": "bullish"|"bearish"|"neutral", "evidence": string}}
  ],
  "entities": {{
    "tickers": [string],
    "sectors": [string],
    "indicators": [string]                // e.g. "CPI", "Fed Funds"
  }},
  "risk_flags": [string]
}}

Document filename: {source}

Content:
{snippet}
"""
        try:
            resp = self.client.messages.create(
                model=Config.LLM_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            # Strip code fences if the model added them despite instructions
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            parsed["method"] = "llm"
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"LLM summary JSON parse failed: {e}")
            return {"method": "llm", "parse_error": str(e), "raw": raw[:500]}
        except anthropic.APIError as e:
            logger.error(f"LLM summary API error: {e}")
            return {"method": "llm", "error": str(e)}


# ═════════════════════════════════════════════════════════
# Formatting helpers (for injection into Ada's context)
# ═════════════════════════════════════════════════════════

def format_summary_text(parsed: ParsedPDF) -> str:
    """Render a ParsedPDF's structured summary as an Ada context block."""
    s = parsed.structured_summary
    lines = [
        f"═══ RESEARCH PDF: {parsed.source} ═══",
        f"Pages: {parsed.page_count} | Extraction: {parsed.extraction_method} | "
        f"Format: {parsed.format_detected}",
        "",
    ]

    if s.get("format") == "S&P Select Industry Dashboard":
        if s.get("as_of"):
            lines.append(f"As of: {s['as_of']}")
        lines.append(f"Sectors tracked: {s.get('sector_count', 0)}")
        lines.append("")
        lines.append("LEADERS (QTD):")
        for r in s.get("leaders", []):
            lines.append(f"  {r['sector']}: QTD {r['qtd_pct']:+.2f}% | MTD {r['mtd_pct']:+.2f}% | 12M {r['twelve_month_pct']:+.2f}%")
        lines.append("LAGGARDS (QTD):")
        for r in s.get("laggards", []):
            lines.append(f"  {r['sector']}: QTD {r['qtd_pct']:+.2f}% | MTD {r['mtd_pct']:+.2f}% | 12M {r['twelve_month_pct']:+.2f}%")
    elif s.get("headline"):
        if s.get("as_of"):
            lines.append(f"As of: {s['as_of']}")
        lines.append(f"Type: {s.get('document_type', 'unknown')}")
        lines.append(f"Headline: {s['headline']}")
        if s.get("key_metrics"):
            lines.append("Key Metrics:")
            for m in s["key_metrics"][:7]:
                lines.append(f"  • {m}")
        if s.get("signals"):
            lines.append("Signals:")
            for sig in s["signals"][:5]:
                lines.append(f"  {sig['direction'].upper()} {sig['topic']}: {sig['evidence']}")
        if s.get("risk_flags"):
            lines.append(f"Risks: {', '.join(s['risk_flags'][:5])}")
    elif s.get("error"):
        lines.append(f"[Summary unavailable: {s['error']}]")

    return "\n".join(lines)
