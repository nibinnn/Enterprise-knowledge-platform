"""
app/core/base/vector_store.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all vector store backends.

The concrete implementation (QdrantVectorStore) is built on Day 7.
This interface is stable — the rest of the pipeline depends on it.

Supported operations:
  - upsert:           store chunks + embeddings
  - search:           dense ANN search
  - keyword_search:   BM25 / sparse search (if the backend supports it)
  - hybrid_search:    dense + keyword fused (Qdrant native)
  - delete:           remove chunks by id
  - delete_document:  remove all chunks for a document
  - health_check:     liveness probe
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.core.models.document import EmbeddedChunk, SearchResult

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):

    # ── Write ──────────────────────────────────────────────────────────────────

    @abstractmethod
    async def upsert(self, chunks: List[EmbeddedChunk]) -> int:
        """
        Insert or update chunks in the vector store.
        Returns the number of chunks successfully upserted.

        Uses the chunk.id as the point id — re-upserting the same id
        is idempotent (overwrites the existing vector + payload).
        """

    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> int:
        """Delete specific chunks by their ids. Returns deleted count."""

    @abstractmethod
    async def delete_document(self, doc_id: str) -> int:
        """
        Delete ALL chunks belonging to a document.
        Used when re-indexing or removing a document.
        Returns deleted count.
        """

    # ── Read ───────────────────────────────────────────────────────────────────

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Dense ANN (approximate nearest-neighbour) search.

        Args:
            query_embedding:  The embedded query vector.
            top_k:            Maximum results to return.
            filters:          Metadata filters (e.g. {"department": "HR"}).
                              Exact syntax is backend-specific; normalise in subclass.
            score_threshold:  Minimum similarity score (0-1). Chunks below this
                              score are discarded even if within top_k.
        Returns:
            List[SearchResult] sorted by score descending.
        """

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Sparse keyword (BM25) search.
        Returns results in the same SearchResult format as `search()`.
        """

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 20,
        dense_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Fused dense + keyword search.
        The implementation may use Qdrant's native hybrid search
        or apply Reciprocal Rank Fusion (RRF) on two separate result lists.
        """

    # ── Admin ──────────────────────────────────────────────────────────────────

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the vector store is reachable and the collection exists."""

    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """Return stats about the collection (vector count, dimensions, etc.)."""

    # ── Utilities (shared across all implementations) ─────────────────────────

    @staticmethod
    def _build_filter_payload(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Normalise a flat key→value filter dict into the backend's expected format.
        Subclasses may override this for backend-specific filter syntax.
        """
        return filters
