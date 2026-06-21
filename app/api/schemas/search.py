"""app/api/schemas/search.py — vector search request/response schemas."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    department: Optional[str] = None
    doc_category: Optional[str] = None
    file_type: Optional[str] = None
    tags: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid")         # dense | keyword | hybrid
    filters: Optional[SearchFilters] = None
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    include_content: bool = True                # return chunk text in results


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    department: Optional[str] = None
    doc_category: Optional[str] = None
    search_type: str = "hybrid"


class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    total_results: int
    mode: str
    latency_ms: float
