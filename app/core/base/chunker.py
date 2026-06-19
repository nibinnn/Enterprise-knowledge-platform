"""
app/core/base/chunker.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all chunking strategies.

Three concrete implementations will be built on Days 4-5:
  - FixedChunker      → splits by token count with overlap
  - RecursiveChunker  → splits on paragraph / sentence / word boundaries
  - SemanticChunker   → splits on embedding-space similarity breakpoints

The ChunkerFactory selects the right strategy per document type or
explicit user request.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.models.document import Chunk, ChunkMetadata, ChunkingStrategy, Document

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    Contract every chunker must fulfil.

    Subclasses implement `_chunk()`. The public `chunk()` method
    handles logging, validation, and metadata injection.
    """

    strategy: ChunkingStrategy  # must be set by each subclass

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Args:
            chunk_size:    Target size per chunk in tokens.
            chunk_overlap: Number of tokens to overlap between consecutive chunks.
                           Overlap ensures no context is lost at chunk boundaries.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a Document into a list of Chunks.

        Returns an empty list (not an error) if the document has no text —
        the ingestion pipeline skips empty documents gracefully.
        """
        text = document.text_for_chunking
        if not text.strip():
            logger.warning("Document '%s' has no text — skipping chunking.", document.filename)
            return []

        logger.info(
            "Chunking '%s' with %s (size=%d, overlap=%d)",
            document.filename, self.__class__.__name__,
            self.chunk_size, self.chunk_overlap,
        )

        raw_chunks = self._chunk(text, document)

        # Inject metadata and assign chunk_index
        enriched: List[Chunk] = []
        for i, chunk in enumerate(raw_chunks):
            chunk.metadata.chunk_index = i
            chunk.metadata.chunking_strategy = self.strategy
            chunk.metadata.char_count = len(chunk.text)
            enriched.append(chunk)

        logger.info(
            "Produced %d chunks from '%s'", len(enriched), document.filename
        )
        return enriched

    # ── To implement ──────────────────────────────────────────────────────────

    @abstractmethod
    def _chunk(self, text: str, document: Document) -> List[Chunk]:
        """
        Core chunking logic.
        Receives the full document text + the Document object (for metadata).
        Returns a list of Chunks WITHOUT chunk_index set (the public
        wrapper handles that).
        """

    # ── Helper shared across all chunkers ─────────────────────────────────────

    def _make_chunk(
        self,
        text: str,
        document: Document,
        page_number: Optional[int] = None,
        section_heading: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> Chunk:
        """
        Convenience factory — creates a Chunk with a pre-populated
        ChunkMetadata from the parent Document.
        """
        meta = ChunkMetadata(
            doc_id=document.id,
            doc_filename=document.filename,
            doc_type=document.doc_type,
            page_number=page_number,
            section_heading=section_heading,
            section_id=section_id,
            chunking_strategy=self.strategy,
            char_count=len(text),
            department=document.metadata.department,
            doc_category=document.metadata.doc_category,
            tags=list(document.metadata.tags),
        )
        return Chunk(text=text.strip(), metadata=meta)


# ─────────────────────────── Factory ─────────────────────────────────────────

class ChunkerFactory:
    """
    Returns the appropriate chunker for a given strategy name.
    Concrete chunkers are imported lazily (on Day 4/5) to keep
    this file free of implementation dependencies.
    """

    @staticmethod
    def get(
        strategy: ChunkingStrategy | str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> BaseChunker:
        """
        Args:
            strategy:      "fixed" | "recursive" | "semantic"
            chunk_size:    Target chunk size in tokens.
            chunk_overlap: Overlap in tokens.

        Returns:
            A configured BaseChunker subclass instance.

        Raises:
            ValueError: if the strategy is not recognised.
        """
        strategy = ChunkingStrategy(strategy)

        # Lazy imports — concrete classes added on Days 4-5
        if strategy == ChunkingStrategy.FIXED:
            from app.chunkers.fixed_chunker import FixedChunker
            return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if strategy == ChunkingStrategy.RECURSIVE:
            from app.chunkers.recursive_chunker import RecursiveChunker
            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if strategy == ChunkingStrategy.SEMANTIC:
            from app.chunkers.semantic_chunker import SemanticChunker
            return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        raise ValueError(f"Unknown chunking strategy: '{strategy}'")
