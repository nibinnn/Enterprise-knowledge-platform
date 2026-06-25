"""
app/retrieval/retriever.py
─────────────────────────────────────────────────────────────────────────────
HybridRetriever — the main entry point for the search layer.

Pipeline per query:
  1. Optional query rewriting (expand abbreviations, fix typos)
  2. Embed the query
  3. Hybrid search (dense + keyword → RRF fusion) via QdrantVectorStore
  4. Re-rank with cross-encoder
  5. Return List[RetrievedChunk]
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.base.retriever import BaseRetriever
from app.core.models.document import RetrievedChunk, SearchResult
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridRetriever(BaseRetriever):
    """
    Production retriever wiring together embedder, vector store, and reranker.

    Args:
        embedder:    BaseEmbedder instance.
        vector_store: QdrantVectorStore instance.
        reranker:    Reranker instance (default: CrossEncoderReranker).
        search_mode: "dense" | "keyword" | "hybrid"
        dense_weight: Weight for dense results in RRF (default 0.7).
        keyword_weight: Weight for keyword results in RRF (default 0.3).
    """

    def __init__(
        self,
        embedder=None,
        vector_store=None,
        reranker=None,
        search_mode:    str   = "hybrid",
        dense_weight:   float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self._embedder      = embedder
        self._vector_store  = vector_store
        self._reranker      = reranker
        self._search_mode   = search_mode
        self._dense_weight  = dense_weight
        self._keyword_weight = keyword_weight

    async def retrieve(
        self,
        query:   str,
        top_k:   int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        # 1. Rewrite query
        query_text = await self.rewrite_query(query)

        # 2. Embed query
        embedder  = self._get_embedder()
        query_vec = await embedder.embed_query(query_text)

        # 3. Search
        vs = self._get_vector_store()
        mode = self._search_mode

        if mode == "dense":
            raw = await vs.search(query_vec, top_k, filters)
        elif mode == "keyword":
            raw = await vs.keyword_search(query_text, top_k, filters)
        else:
            raw = await vs.hybrid_search(
                query_text, query_vec, top_k,
                self._dense_weight, self._keyword_weight, filters,
            )

        retrieved = self._to_retrieved_chunks(raw)

        # 4. Rerank
        reranker = self._get_reranker()
        reranked = await reranker.rerank(query_text, retrieved, top_k)

        logger.info(
            "Retrieved %d chunks for query='%s…' (mode=%s)",
            len(reranked), query[:60], mode,
        )
        return reranked

    async def rewrite_query(self, query: str) -> str:
        """
        Light query normalisation: strip excess whitespace, lowercase.
        More sophisticated rewriting (HyDE, query expansion) can be added here.
        """
        return " ".join(query.split())

    # ── Lazy dependency resolution ────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from app.core.base.embedder import EmbedderFactory
            self._embedder = EmbedderFactory.get()
        return self._embedder

    def _get_vector_store(self):
        if self._vector_store is None:
            from app.vector_store.qdrant_store import QdrantVectorStore
            self._vector_store = QdrantVectorStore()
        return self._vector_store

    def _get_reranker(self):
        if self._reranker is None:
            from app.retrieval.reranker import NoOpReranker
            self._reranker = NoOpReranker()   # swap for CrossEncoderReranker in prod
        return self._reranker
