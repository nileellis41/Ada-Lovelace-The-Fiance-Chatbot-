"""
Market Bridge — Semantic Chunker
=================================
Embeddings-based chunking for unstructured documents like 8-K earnings
transcripts. Uses sentence-transformers to detect topic boundaries via
cosine similarity drop-off between adjacent text windows.

Strategy:
1. Split text into sentences
2. Compute embeddings for sliding windows of sentences
3. Detect breakpoints where cosine similarity drops below threshold
4. Merge adjacent segments into chunks respecting min/max size constraints
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from market_bridge.config.settings import ChunkingConfig
from market_bridge.utils.helpers import get_logger, normalize_text

logger = get_logger("chunker.semantic")


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    chunk_id: int
    source: str
    chunk_type: str          # "semantic" or "template"
    section_label: str = ""  # e.g., "Item 7 - MD&A"
    start_char: int = 0
    end_char: int = 0
    token_estimate: int = 0  # rough token count
    
    def __post_init__(self):
        self.token_estimate = len(self.text.split()) * 1.3  # rough heuristic


class SemanticChunker:
    """
    Embeddings-based semantic chunker for unstructured financial text.
    Designed for 8-K earnings transcripts where section headers are
    inconsistent or absent.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.semantic_model}")
            self._model = SentenceTransformer(self.config.semantic_model)
        return self._model
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex-based heuristics tuned for financial text."""
        # Handle common abbreviations in financial text
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Inc|Corp|Ltd|Co|vs|etc|approx|est)\.',
                      r'\1<PERIOD>', text)
        text = re.sub(r'\b(\d+)\.\s*(\d+)', r'\1<DECIMAL>\2', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore periods
        sentences = [
            s.replace('<PERIOD>', '.').replace('<DECIMAL>', '.')
            for s in sentences
        ]
        
        # Filter out tiny fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences
    
    def _compute_similarities(self, sentences: List[str], window_size: int = 3) -> np.ndarray:
        """Compute cosine similarity between adjacent sliding windows of sentences."""
        if len(sentences) <= window_size:
            return np.ones(max(len(sentences) - 1, 1))
        
        # Create sliding windows
        windows = []
        for i in range(len(sentences) - window_size + 1):
            window_text = " ".join(sentences[i:i + window_size])
            windows.append(window_text)
        
        # Compute embeddings
        embeddings = self.model.encode(windows, show_progress_bar=False)
        
        # Cosine similarity between adjacent windows
        similarities = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append(float(cos_sim))
        
        return np.array(similarities)
    
    def _find_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """Find indices where topic shifts occur (similarity drops below threshold)."""
        threshold = self.config.semantic_similarity_threshold
        breakpoints = []
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i)
        
        # If no natural breakpoints, use percentile-based approach
        if not breakpoints and len(similarities) > 5:
            dynamic_threshold = np.percentile(similarities, 25)
            breakpoints = [i for i, sim in enumerate(similarities) if sim < dynamic_threshold]
        
        return breakpoints
    
    def _merge_segments(
        self,
        sentences: List[str],
        breakpoints: List[int],
        window_size: int = 3,
    ) -> List[str]:
        """Merge sentences between breakpoints into chunks, respecting size constraints."""
        if not breakpoints:
            # No breakpoints — chunk by size
            return self._chunk_by_size(sentences)
        
        # Convert window-based breakpoints to sentence indices
        split_indices = [bp + window_size // 2 for bp in breakpoints]
        split_indices = sorted(set(split_indices))
        
        # Build segments
        segments = []
        prev = 0
        for idx in split_indices:
            if idx > prev:
                segment = " ".join(sentences[prev:idx])
                segments.append(segment)
            prev = idx
        # Last segment
        if prev < len(sentences):
            segments.append(" ".join(sentences[prev:]))
        
        # Enforce size constraints
        final_segments = []
        for seg in segments:
            if len(seg) > self.config.semantic_max_chunk_size:
                # Sub-chunk oversized segments
                sub_sents = self._split_sentences(seg)
                final_segments.extend(self._chunk_by_size(sub_sents))
            elif len(seg) < self.config.semantic_min_chunk_size and final_segments:
                # Merge undersized with previous
                final_segments[-1] += " " + seg
            else:
                final_segments.append(seg)
        
        return final_segments
    
    def _chunk_by_size(self, sentences: List[str]) -> List[str]:
        """Fallback: chunk by character count with overlap."""
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.config.semantic_max_chunk_size and current:
                chunks.append(" ".join(current))
                # Keep last sentence for overlap
                overlap_sents = current[-1:] if current else []
                current = overlap_sents
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += sent_len
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    def chunk(
        self,
        text: str,
        source: str = "8-K",
        filing_ticker: str = "",
    ) -> List[Chunk]:
        """
        Chunk text using semantic similarity breakpoints.
        
        Args:
            text: Raw document text (cleaned)
            source: Source identifier for metadata
            filing_ticker: Ticker symbol for labeling
        
        Returns:
            List of Chunk objects
        """
        text = normalize_text(text)
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            return [Chunk(
                text=text,
                chunk_id=0,
                source=source,
                chunk_type="semantic",
                section_label=f"{filing_ticker} earnings transcript",
            )]
        
        logger.info(f"Semantic chunking {len(sentences)} sentences from {source}")
        
        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)
        segments = self._merge_segments(sentences, breakpoints)
        
        chunks = []
        offset = 0
        for i, seg_text in enumerate(segments):
            start = text.find(seg_text[:50], offset)
            if start == -1:
                start = offset
            end = start + len(seg_text)
            
            chunks.append(Chunk(
                text=seg_text,
                chunk_id=i,
                source=source,
                chunk_type="semantic",
                section_label=f"{filing_ticker} earnings — segment {i+1}/{len(segments)}",
                start_char=start,
                end_char=end,
            ))
            offset = max(offset, start + 1)
        
        logger.info(f"Produced {len(chunks)} semantic chunks (avg {sum(len(c.text) for c in chunks)//max(len(chunks),1)} chars)")
        return chunks
