"""
app/services/search.py
Stub until Day 15 (Qdrant vector store + hybrid search).
Interface is stable — routes call this, not the vector store directly.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from app.api.schemas.search import SearchRequest, SearchResponse, ChunkResult


class SearchService:

    async def search(self, request: SearchRequest, db) -> SearchResponse:
        """
        Search the knowledge base.
        Stub returns an empty result set.
        Day 15 replaces this body with:
            embedder → embed query
            vector_store → hybrid_search()
            reranker → rerank()
        """
        start = time.perf_counter()

        # TODO (Day 15): real search implementation
        results: List[ChunkResult] = []

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=0,
            mode=request.mode,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
