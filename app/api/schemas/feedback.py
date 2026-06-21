"""app/api/schemas/feedback.py — user feedback schemas."""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: int = Field(..., ge=1, le=5)
    correction: Optional[str] = Field(default=None, max_length=5000)
    bad_citation_ids: List[str] = Field(default_factory=list)
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    answer_id: str
    rating: int
    created_at: datetime

    model_config = {"from_attributes": True}
