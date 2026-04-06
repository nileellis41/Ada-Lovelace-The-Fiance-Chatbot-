"""
RAG Pipeline
─────────────
ChromaDB-backed retrieval-augmented generation for finance documents.
Supports ingesting filings, research notes, and arbitrary text.
"""
import hashlib
import logging
import re
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import Config

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


class VectorStore:
    """ChromaDB wrapper with finance-specific ingestion helpers."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.CHROMA_COLLECTION

        self.client = chromadb.Client(Settings(
            is_persistent=True,
            persist_directory=self.persist_dir,
            anonymized_telemetry=False,
        ))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.collection_name}' "
            f"loaded with {self.collection.count()} documents"
        )

    # ── Ingest ───────────────────────────────────────────
    def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """Chunk and store a document. Returns number of chunks added."""
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        metadata = metadata or {}

        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            return 0

        ids = [f"{_doc_id(c)}_{i}" for i, c in enumerate(chunks)]
        metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"Ingested {len(chunks)} chunks (source: {metadata.get('source', 'unknown')})")
        return len(chunks)

    def ingest_filing(self, ticker: str, form_type: str, text: str, date: str = "") -> int:
        """Convenience: ingest an SEC filing with structured metadata."""
        return self.ingest_text(
            text,
            metadata={
                "source": "edgar",
                "ticker": ticker.upper(),
                "form_type": form_type,
                "filing_date": date,
            },
        )

    # ── Query ────────────────────────────────────────────
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a question."""
        top_k = top_k or Config.TOP_K_RESULTS
        kwargs = {
            "query_texts": [question],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return docs

    def format_context(self, question: str, top_k: Optional[int] = None) -> str:
        """Query and format results as LLM context block."""
        docs = self.query(question, top_k)
        if not docs:
            return ""

        lines = ["═══ RETRIEVED CONTEXT (RAG) ═══"]
        for i, doc in enumerate(docs, 1):
            meta = doc["metadata"]
            source = meta.get("source", "unknown")
            ticker = meta.get("ticker", "")
            dist = f"{doc['distance']:.3f}" if doc["distance"] is not None else "N/A"
            header = f"[{i}] source={source}"
            if ticker:
                header += f" ticker={ticker}"
            header += f" distance={dist}"
            lines.append(header)
            lines.append(doc["text"][:500])
            lines.append("")
        return "\n".join(lines)

    # ── Admin ────────────────────────────────────────────
    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Delete all documents in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")
