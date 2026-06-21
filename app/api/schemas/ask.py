"""app/api/schemas/ask.py — RAG question answering schemas."""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.api.schemas.search import SearchFilters


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[SearchFilters] = None
    top_k: int = Field(default=20, ge=1, le=50)     # chunks retrieved
    rerank_top_n: int = Field(default=5, ge=1, le=20)  # chunks after rerank
    mode: str = Field(default="rag")                # rag | agent
    stream: bool = False
    include_sources: bool = True
    language: Optional[str] = None                  # answer language hint


class CitationOut(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    excerpt: str


class AskResponse(BaseModel):
    answer_id: str
    question: str
    answer: str
    citations: List[CitationOut] = Field(default_factory=list)
    confidence: Optional[float] = None
    model_used: str
    retrieval_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentStepOut(BaseModel):
    """One reasoning step taken by the agent (for agentic research mode)."""
    step: int
    tool_name: str
    tool_input: str
    tool_output: str
    latency_ms: float


class AgentAskResponse(AskResponse):
    """Extended response when mode='agent' — includes reasoning trace."""
    steps: List[AgentStepOut] = Field(default_factory=list)
    total_steps: int = 0
