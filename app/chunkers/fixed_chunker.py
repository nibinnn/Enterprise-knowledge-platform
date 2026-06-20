"""
app/chunkers/fixed_chunker.py
─────────────────────────────────────────────────────────────────────────────
Fixed-size chunker — sliding token window with configurable overlap.

Best for:
  - Documents with no clear structure (raw text dumps, logs)
  - When downstream embedding model has a hard token limit
  - When you want guaranteed uniform chunk sizes for benchmarking

Not ideal for:
  - Structured documents (use RecursiveChunker instead — it respects
    paragraph and sentence boundaries)
  - Semantic search (chunks split mid-sentence have less coherent embeddings)

Algorithm:
  1. Tokenise the full document text with tiktoken (or word-count fallback)
  2. Slide a window of `chunk_size` tokens, stepping by `chunk_size - overlap`
  3. Decode each window back to a string
  4. Build one Chunk per window with full provenance metadata
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.chunkers.token_counter import TokenCounter
from app.core.base.chunker import BaseChunker
from app.core.models.document import Chunk, ChunkingStrategy, Document

logger = logging.getLogger(__name__)


class FixedChunker(BaseChunker):
    """
    Splits a Document into fixed-size token windows.

    Args:
        chunk_size:    Target number of tokens per chunk  (default 512).
        chunk_overlap: Tokens shared between adjacent chunks (default 64).
        tiktoken_model: tiktoken encoding to use (default "cl100k_base").
    """

    strategy = ChunkingStrategy.FIXED

    def __init__(
        self,
        chunk_size:     int = 512,
        chunk_overlap:  int = 64,
        tiktoken_model: str = "cl100k_base",
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._tc = TokenCounter(model=tiktoken_model)

    # ── BaseChunker interface ─────────────────────────────────────────────────

    def _chunk(self, text: str, document: Document) -> List[Chunk]:
        raw_chunks = self._tc.chunk_by_tokens(
            text, self.chunk_size, self.chunk_overlap
        )

        chunks: List[Chunk] = []
        for raw_text in raw_chunks:
            stripped = raw_text.strip()
            if not stripped:
                continue

            chunk = self._make_chunk(
                text=stripped,
                document=document,
                # FixedChunker has no page/section awareness — those come
                # from the document metadata only
                page_number=None,
                section_heading=None,
            )
            chunk.metadata.token_count = self._tc.count(stripped)
            chunks.append(chunk)

        return chunks

    def count_tokens(self, text: str) -> int:
        """Expose token counting for external callers (e.g. the eval framework)."""
        return self._tc.count(text)
