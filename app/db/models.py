"""
app/db/models.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy ORM models for all persisted entities.

Tables:
  documents   — one row per ingested file
  chunks      — one row per text chunk (child of documents)
  jobs        — async ingestion / processing tasks
  feedback    — user ratings & corrections on answers
  eval_runs   — evaluation framework results (Day 20)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────── Documents ───────────────────────────────────────

class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)  # pdf, docx …
    file_path: Mapped[str] = mapped_column(String(1024), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Parsed content
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Filterable metadata (denormalised for fast filter queries)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    author: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    department: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    doc_category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    doc_created_date: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Flexible extra metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True, default=list)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    chunks: Mapped[List["ChunkModel"]] = relationship(
        "ChunkModel", back_populates="document",
        cascade="all, delete-orphan", passive_deletes=True,
    )
    jobs: Mapped[List["JobModel"]] = relationship(
        "JobModel", back_populates="document",
        cascade="all, delete-orphan", passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id!r} filename={self.filename!r} status={self.status!r}>"


# ─────────────────────────── Chunks ──────────────────────────────────────────

class ChunkModel(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    doc_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Chunking provenance
    chunking_strategy: Mapped[str] = mapped_column(
        String(32), nullable=False, default="recursive"
    )

    # Source location
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    section_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Embedding status
    is_embedded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Filter fields (inherited from parent document for fast vector-store filtering)
    department: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    doc_category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk id={self.id!r} doc_id={self.doc_id!r} index={self.chunk_index}>"


# ─────────────────────────── Jobs ────────────────────────────────────────────

class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    doc_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    job_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="ingest"
        # ingest | re_embed | delete | re_chunk
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True
        # pending | running | success | failed | retrying
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    progress_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationship
    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} type={self.job_type!r} status={self.status!r}>"


# ─────────────────────────── Feedback ────────────────────────────────────────

class FeedbackModel(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)       # 1-5
    correction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bad_citation_ids: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    extra_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Feedback id={self.id!r} rating={self.rating} answer_id={self.answer_id!r}>"


# ─────────────────────────── Eval Runs (Day 20) ───────────────────────────────

class EvalRunModel(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    run_name: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(256), nullable=False)

    # RAGAS metric scores (0.0 – 1.0)
    faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Config snapshot
    llm_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    chunking_strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    search_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    rerank_top_n: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    num_questions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    results_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<EvalRun id={self.id!r} name={self.run_name!r}>"
