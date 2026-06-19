"""
app/core/base/retriever.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for the retriever layer.

The retriever sits between raw search (VectorStore) and context
assembly (ContextBuilder). Its responsibilities:
  - Accept a natural-language query
  - Optionally rewrite / expand the query before search
  - Call the VectorStore (dense / keyword / hybrid)
  - Return RetrievedChunk objects ready for re-ranking

The concrete implementation (HybridRetriever) is built on Day 11.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.core.models.document import RetrievedChunk, SearchResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:   Natural-language question / search string.
            top_k:   Number of chunks to return (before re-ranking).
            filters: Optional metadata filters (department, doc_type …).

        Returns:
            List of RetrievedChunk, sorted by score descending.
            The `rank` field is set sequentially (1 = best).
        """

    # ── Optional hooks (subclasses may override) ──────────────────────────────

    async def rewrite_query(self, query: str) -> str:
        """
        Optionally rewrite or expand the user query before search.
        Default: return the query unchanged.
        Override on Day 11 to add HyDE or query expansion.
        """
        return query

    @staticmethod
    def _to_retrieved_chunks(results: List[SearchResult]) -> List[RetrievedChunk]:
        """Convert raw SearchResult objects into RetrievedChunk objects."""
        retrieved = []
        for rank, result in enumerate(results, start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=result.score,
                    original_score=result.score,
                    rerank_score=None,
                    metadata=result.metadata,
                    rank=rank,
                )
            )
        return retrieved
