"""app/api/schemas/common.py — shared envelope, pagination, and error schemas."""
from __future__ import annotations
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar("T")


class Meta(BaseModel):
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationMeta(Meta):
    total: int
    page: int
    page_size: int
    total_pages: int


class APIResponse(BaseModel, Generic[T]):
    """Standard envelope wrapping every successful response."""
    success: bool = True
    data: T
    meta: Optional[Meta] = None


class PaginatedResponse(BaseModel, Generic[T]):
    success: bool = True
    data: List[T]
    meta: PaginationMeta


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    meta: Optional[Meta] = None


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
