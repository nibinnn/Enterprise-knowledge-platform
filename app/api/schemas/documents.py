"""app/api/schemas/documents.py — document ingestion request/response schemas."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentUploadResponse(BaseModel):
    """Returned immediately after a file is accepted for ingestion."""
    document_id: str
    job_id: str
    filename: str
    file_type: str
    status: str = "pending"
    message: str = "Document accepted for ingestion."


class DocumentMetadataIn(BaseModel):
    """Optional metadata the caller can supply at upload time."""
    title: Optional[str] = None
    department: Optional[str] = None
    doc_category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentOut(BaseModel):
    """Full document representation returned by GET /documents/{id}."""
    id: str
    filename: str
    file_type: str
    status: str
    title: Optional[str] = None
    author: Optional[str] = None
    department: Optional[str] = None
    doc_category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    indexed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentListOut(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    title: Optional[str] = None
    department: Optional[str] = None
    chunk_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentStatusOut(BaseModel):
    document_id: str
    job_id: Optional[str] = None
    status: str
    progress_pct: Optional[float] = None
    error: Optional[str] = None
    chunk_count: int = 0
    indexed_at: Optional[datetime] = None


class DocumentFilterParams(BaseModel):
    status: Optional[str] = None
    file_type: Optional[str] = None
    department: Optional[str] = None
    doc_category: Optional[str] = None
    search: Optional[str] = None       # filename / title substring search

    @field_validator("status")
    @classmethod
    def valid_status(cls, v):
        allowed = {"pending", "processing", "indexed", "failed", "archived"}
        if v and v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v
