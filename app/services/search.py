"""
app/services/search.py
Real search: HybridRetriever (embed → Qdrant → rerank) mapped to SearchResponse.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.api.schemas.search import ChunkResult, SearchRequest, SearchResponse


def _filters_to_dict(filters) -> Optional[Dict[str, Any]]:
    if filters is None:
        return None
    d: Dict[str, Any] = {}
    if filters.department:
        d["department"] = filters.department
    if filters.doc_category:
        d["doc_category"] = filters.doc_category
    if filters.file_type:
        d["doc_type"] = filters.file_type
    if filters.tags:
        d["tags"] = filters.tags
    if filters.document_ids:
        d["doc_id"] = filters.document_ids
    return d or None


class SearchService:

    async def search(self, request: SearchRequest, db) -> SearchResponse:
        from app.retrieval.retriever import HybridRetriever

        start = time.perf_counter()
        retriever = HybridRetriever(search_mode=request.mode)
        chunks = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=_filters_to_dict(request.filters),
        )

        results: List[ChunkResult] = []
        for c in chunks:
            score = round(c.score, 4)
            if request.score_threshold is not None and score < request.score_threshold:
                continue
            results.append(
                ChunkResult(
                    chunk_id=c.chunk_id,
                    document_id=c.metadata.doc_id,
                    filename=c.metadata.doc_filename,
                    text=c.text if request.include_content else "",
                    score=score,
                    page_number=c.metadata.page_number,
                    section_heading=c.metadata.section_heading,
                    department=c.metadata.department,
                    doc_category=c.metadata.doc_category,
                    search_type=request.mode,
                )
            )

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            mode=request.mode,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
