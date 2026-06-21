"""app/api/routes/search.py — knowledge base search endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db, get_search_service
from app.api.schemas.auth import CurrentUser
from app.api.schemas.common import APIResponse
from app.api.schemas.search import SearchRequest, SearchResponse
from app.services.search import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/",
    response_model=APIResponse[SearchResponse],
    summary="Search the knowledge base",
)
async def search(
    body:         SearchRequest,
    current_user: CurrentUser     = Depends(get_current_user),
    db:           AsyncSession    = Depends(get_db),
    svc:          SearchService   = Depends(get_search_service),
):
    """
    Search indexed documents using dense, keyword, or hybrid (default) mode.

    - **dense**:   vector similarity search (best for semantic queries)
    - **keyword**: BM25 full-text search (best for exact term matching)
    - **hybrid**:  Reciprocal Rank Fusion of both (best overall)

    Supports metadata filters: department, doc_category, file_type, tags.
    """
    result = await svc.search(body, db)
    return APIResponse(data=result)


@router.post(
    "/similar",
    response_model=APIResponse[SearchResponse],
    summary="Find chunks similar to a given chunk",
)
async def find_similar(
    chunk_id:     str,
    top_k:        int = 5,
    current_user: CurrentUser   = Depends(get_current_user),
    db:           AsyncSession  = Depends(get_db),
    svc:          SearchService = Depends(get_search_service),
):
    """
    Return the top-k chunks most similar to a given chunk_id.
    Useful for 'read more like this' features.
    Stub — real implementation queries Qdrant by vector on Day 15.
    """
    # TODO (Day 15): look up chunk embedding from Qdrant, then similarity search
    return APIResponse(
        data=SearchResponse(
            query=f"similar_to:{chunk_id}",
            results=[],
            total_results=0,
            mode="dense",
            latency_ms=0.0,
        )
    )
