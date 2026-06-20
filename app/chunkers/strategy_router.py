"""
app/chunkers/strategy_router.py
─────────────────────────────────────────────────────────────────────────────
Automatic chunking strategy selector.

Rules (in priority order):

  Explicit override
    If the caller passes strategy="fixed"|"recursive"|"semantic", use that.

  Document-type rules
    .md / .markdown     → Recursive  (headings are already semantic breaks)
    .html               → Recursive  (DOM structure gives good boundaries)
    .docx               → Recursive  (heading styles give good boundaries)
    .txt                → Recursive  (paragraph breaks respected)
    .pdf (structured)   → Recursive  (sections detected by PDFParser)
    .pdf (flat/OCR)     → Fixed      (no reliable structure, uniform chunks)
    Any                 → Recursive  (safe default)

  Content-based overrides
    If a document has NO sections and is very long → Fixed
    If the caller explicitly requests semantic      → Semantic

The router also exposes a quick comparison helper used by the Day 5 eval
script to measure chunk quality across strategies.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.core.base.chunker import BaseChunker
from app.core.models.document import Chunk, ChunkingStrategy, Document, DocumentType

logger = logging.getLogger(__name__)

# Document-type → preferred strategy
_TYPE_STRATEGY_MAP: Dict[DocumentType, ChunkingStrategy] = {
    DocumentType.PDF:     ChunkingStrategy.RECURSIVE,
    DocumentType.DOCX:    ChunkingStrategy.RECURSIVE,
    DocumentType.MD:      ChunkingStrategy.RECURSIVE,
    DocumentType.HTML:    ChunkingStrategy.RECURSIVE,
    DocumentType.TXT:     ChunkingStrategy.RECURSIVE,
    DocumentType.UNKNOWN: ChunkingStrategy.RECURSIVE,
}

# If a PDF has fewer than this many sections it's treated as flat → Fixed
_PDF_MIN_SECTIONS_FOR_RECURSIVE = 3

# If flat text exceeds this token count, prefer Fixed over Recursive
# (very long flat text causes deep recursion)
_FLAT_TEXT_TOKEN_THRESHOLD = 50_000


class ChunkingStrategyRouter:
    """
    Selects and executes the most appropriate chunking strategy.

    Usage:
        router = ChunkingStrategyRouter(chunk_size=512, chunk_overlap=64)
        chunks = router.chunk(document)

        # Force a specific strategy:
        chunks = router.chunk(document, strategy="semantic")
    """

    def __init__(
        self,
        chunk_size:           int   = 512,
        chunk_overlap:        int   = 64,
        breakpoint_threshold: float = 0.85,
        tiktoken_model:       str   = "cl100k_base",
    ):
        self.chunk_size           = chunk_size
        self.chunk_overlap        = chunk_overlap
        self.breakpoint_threshold = breakpoint_threshold
        self.tiktoken_model       = tiktoken_model

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        document: Document,
        strategy: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk *document* using the auto-selected or explicitly requested strategy.

        Args:
            document: Parsed Document object.
            strategy: Optional override — "fixed" | "recursive" | "semantic".

        Returns:
            List[Chunk] with metadata populated.
        """
        selected = self._select_strategy(document, strategy)
        chunker  = self._build_chunker(selected)

        logger.info(
            "Chunking '%s' with strategy=%s (doc_type=%s, sections=%d)",
            document.filename, selected.value,
            document.doc_type.value, len(document.sections),
        )

        return chunker.chunk(document)

    def compare_strategies(
        self,
        document: Document,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, List[Chunk]]:
        """
        Chunk the same document with multiple strategies and return a
        comparison dict.  Used by the Day 5 evaluation script.

        Args:
            document:   Document to chunk.
            strategies: List of strategy names to compare.
                        Default: ["fixed", "recursive"]
                        ("semantic" excluded by default — needs embedder)

        Returns:
            {"fixed": [...], "recursive": [...]}
        """
        strategies = strategies or ["fixed", "recursive"]
        results: Dict[str, List[Chunk]] = {}
        for s in strategies:
            try:
                results[s] = self.chunk(document, strategy=s)
            except Exception as exc:
                logger.warning("Strategy '%s' failed: %s", s, exc)
                results[s] = []
        return results

    def summarise_comparison(
        self,
        comparison: Dict[str, List[Chunk]],
    ) -> Dict[str, dict]:
        """
        Return a stats summary for each strategy in a comparison dict.

        Returns something like:
            {
                "fixed":     {"count": 42, "avg_tokens": 490, "min": 12, "max": 512},
                "recursive": {"count": 38, "avg_tokens": 445, "min": 80, "max": 512},
            }
        """
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter(self.tiktoken_model)

        summary: Dict[str, dict] = {}
        for strategy, chunks in comparison.items():
            if not chunks:
                summary[strategy] = {"count": 0}
                continue
            token_counts = [
                c.metadata.token_count or tc.count(c.text)
                for c in chunks
            ]
            summary[strategy] = {
                "count":      len(chunks),
                "avg_tokens": round(sum(token_counts) / len(token_counts)),
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
            }
        return summary

    # ── Strategy selection ────────────────────────────────────────────────────

    def _select_strategy(
        self,
        document: Document,
        override: Optional[str],
    ) -> ChunkingStrategy:
        # 1. Explicit override wins
        if override:
            return ChunkingStrategy(override)

        # 2. Document-type lookup
        base = _TYPE_STRATEGY_MAP.get(document.doc_type, ChunkingStrategy.RECURSIVE)

        # 3. Content-based adjustment
        if base == ChunkingStrategy.RECURSIVE:
            # PDF with very few sections → fall back to Fixed
            if (
                document.doc_type == DocumentType.PDF
                and len(document.sections) < _PDF_MIN_SECTIONS_FOR_RECURSIVE
            ):
                logger.debug(
                    "PDF '%s' has %d section(s) — using Fixed chunker",
                    document.filename, len(document.sections)
                )
                return ChunkingStrategy.FIXED

            # Very long flat document → Fixed avoids deep recursion
            if (
                not document.has_sections
                and len(document.raw_text) > _FLAT_TEXT_TOKEN_THRESHOLD * 4  # rough char estimate
            ):
                return ChunkingStrategy.FIXED

        return base

    # ── Chunker factory ───────────────────────────────────────────────────────

    def _build_chunker(self, strategy: ChunkingStrategy) -> BaseChunker:
        if strategy == ChunkingStrategy.FIXED:
            from app.chunkers.fixed_chunker import FixedChunker
            return FixedChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                tiktoken_model=self.tiktoken_model,
            )
        if strategy == ChunkingStrategy.RECURSIVE:
            from app.chunkers.recursive_chunker import RecursiveChunker
            return RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                section_aware=True,
                tiktoken_model=self.tiktoken_model,
            )
        if strategy == ChunkingStrategy.SEMANTIC:
            from app.chunkers.semantic_chunker import SemanticChunker
            return SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                breakpoint_threshold=self.breakpoint_threshold,
                tiktoken_model=self.tiktoken_model,
            )
        raise ValueError(f"Unknown chunking strategy: {strategy}")
