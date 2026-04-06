"""
Market Bridge — Template Chunker
==================================
Template-based chunking for structured SEC filings (10-K, 10-Q).

Since sec-api.io's ExtractorApi already returns individual sections 
by Item number, this chunker focuses on:
  1. Sub-chunking oversized sections into digestible pieces
  2. Maintaining section labels and metadata through the pipeline
  3. Adding overlap between sub-chunks for context continuity

This is the complement to SemanticChunker — template handles structured
documents, semantic handles unstructured narratives (8-K transcripts).
"""

import re
from typing import Dict, List, Optional, Tuple

from market_bridge.config.settings import ChunkingConfig
from market_bridge.chunking.semantic_chunker import Chunk
from market_bridge.utils.helpers import get_logger, normalize_text

logger = get_logger("chunker.template")


class TemplateChunker:
    """
    Template-based chunker for pre-extracted SEC filing sections.
    
    Takes the output of sec-api ExtractorApi (Dict[section_label, text])
    and produces Chunk objects with proper metadata. Oversized sections
    are sub-chunked on paragraph boundaries with configurable overlap.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    def _sub_chunk_section(
        self,
        text: str,
        section_label: str,
        max_size: int,
        overlap: int,
    ) -> List[Tuple[str, str]]:
        """
        Break an oversized section into sub-chunks on paragraph boundaries.
        
        Returns:
            List of (sub_label, text) tuples
        """
        if len(text) <= max_size:
            return [(section_label, text)]
        
        # Split on double-newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\n+", text)
        
        # If paragraphs are still too large, split on single newlines
        expanded = []
        for para in paragraphs:
            if len(para) > max_size:
                sub_paras = re.split(r"\n", para)
                expanded.extend(sub_paras)
            else:
                expanded.append(para)
        paragraphs = [p for p in expanded if len(p.strip()) > 20]
        
        sub_chunks = []
        current = []
        current_len = 0
        chunk_num = 1
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_len + para_len > max_size and current:
                chunk_text = "\n\n".join(current)
                sub_chunks.append((
                    f"{section_label} [part {chunk_num}]",
                    chunk_text,
                ))
                chunk_num += 1
                
                # Overlap: keep last paragraph(s) up to overlap chars
                overlap_paras = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > overlap:
                        break
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                
                current = overlap_paras
                current_len = overlap_len
            
            current.append(para)
            current_len += para_len
        
        if current:
            sub_chunks.append((
                f"{section_label} [part {chunk_num}]",
                "\n\n".join(current),
            ))
        
        return sub_chunks
    
    def chunk(
        self,
        sections: Dict[str, str],
        source: str = "10-K",
        filing_ticker: str = "",
    ) -> List[Chunk]:
        """
        Chunk pre-extracted filing sections into Chunk objects.
        
        Args:
            sections: Dict mapping section labels to text content
                      (output of SecApiClient.extract_all_sections)
            source: Filing type identifier
            filing_ticker: Ticker symbol for metadata
        
        Returns:
            List of Chunk objects ready for LLM synthesis
        """
        all_chunks = []
        chunk_id = 0
        
        for section_label, text in sections.items():
            text = normalize_text(text)
            
            if len(text) < 50:
                logger.debug(f"Skipping short section: {section_label}")
                continue
            
            # Sub-chunk if needed
            sub_chunks = self._sub_chunk_section(
                text,
                section_label,
                max_size=self.config.template_fallback_chunk_size,
                overlap=self.config.template_overlap,
            )
            
            for sub_label, sub_text in sub_chunks:
                chunk = Chunk(
                    text=sub_text,
                    chunk_id=chunk_id,
                    source=source,
                    chunk_type="template",
                    section_label=f"{filing_ticker} {sub_label}",
                    start_char=0,
                    end_char=len(sub_text),
                )
                all_chunks.append(chunk)
                chunk_id += 1
        
        logger.info(
            f"Template chunking: {len(sections)} sections → "
            f"{len(all_chunks)} chunks "
            f"(avg {sum(len(c.text) for c in all_chunks) // max(len(all_chunks), 1)} chars)"
        )
        return all_chunks
    
    def chunk_with_context_window(
        self,
        sections: Dict[str, str],
        max_total_tokens: int = 80000,
        source: str = "10-K",
        filing_ticker: str = "",
    ) -> List[Chunk]:
        """
        Chunk sections while respecting a total token budget.
        Prioritizes high-value sections (MD&A, Risk Factors) and
        truncates lower-priority sections if budget is exceeded.
        
        Args:
            sections: Dict mapping section labels to text
            max_total_tokens: Maximum total tokens across all chunks
            source: Filing type
            filing_ticker: Ticker symbol
        
        Returns:
            List of Chunk objects within token budget
        """
        # Priority ordering (highest first)
        priority_keywords = [
            "MD&A", "Risk Factors", "Results of Operations",
            "Financial Statements", "Quantitative",
            "Business", "Controls", "Legal",
        ]
        
        def section_priority(label: str) -> int:
            for i, kw in enumerate(priority_keywords):
                if kw.lower() in label.lower():
                    return i
            return len(priority_keywords)
        
        # Sort sections by priority
        sorted_sections = dict(
            sorted(sections.items(), key=lambda x: section_priority(x[0]))
        )
        
        # Chunk with budget tracking
        all_chunks = []
        total_tokens = 0
        chunk_id = 0
        
        for section_label, text in sorted_sections.items():
            text = normalize_text(text)
            estimated_tokens = len(text.split()) * 1.3
            
            if total_tokens + estimated_tokens > max_total_tokens:
                # Truncate this section to fit remaining budget
                remaining = max_total_tokens - total_tokens
                if remaining < 200:
                    logger.info(f"Token budget exhausted — skipping {section_label}")
                    continue
                # Truncate to approximately remaining tokens
                char_limit = int(remaining / 1.3 * 4.5)
                text = text[:char_limit] + "\n\n[... truncated for context window ...]"
            
            sub_chunks = self._sub_chunk_section(
                text,
                section_label,
                max_size=self.config.template_fallback_chunk_size,
                overlap=self.config.template_overlap,
            )
            
            for sub_label, sub_text in sub_chunks:
                chunk = Chunk(
                    text=sub_text,
                    chunk_id=chunk_id,
                    source=source,
                    chunk_type="template",
                    section_label=f"{filing_ticker} {sub_label}",
                )
                all_chunks.append(chunk)
                total_tokens += len(sub_text.split()) * 1.3
                chunk_id += 1
        
        logger.info(
            f"Budget-aware chunking: {len(all_chunks)} chunks, "
            f"~{int(total_tokens)} tokens (budget: {max_total_tokens})"
        )
        return all_chunks
