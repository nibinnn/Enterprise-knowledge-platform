"""
app/core/models/document.py
─────────────────────────────────────────────────────────────────────────────
Core domain models. These are pure Python dataclasses / Pydantic models —
they have NO database or framework dependencies, making them safe to import
anywhere (parsers, chunkers, embedders, retrieval, agents, tests).

Data flow:
  Raw file
    → Parser → Document
    → Chunker → List[Chunk]
    → Embedder → List[EmbeddedChunk]
    → VectorStore (upsert)
    → Retriever → List[RetrievedChunk]
    → ContextBuilder → QueryContext
    → LLM → Answer
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────── Enums ───────────────────────────────────────────

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "markdown"
    HTML = "html"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    PENDING = "pending"           # uploaded, not yet ingested
    PROCESSING = "processing"     # ingestion in progress
    INDEXED = "indexed"           # successfully in vector store
    FAILED = "failed"             # ingestion failed
    ARCHIVED = "archived"         # soft-deleted


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# ─────────────────────────── Document ────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """
    Flexible metadata bag. Parsers populate what they can;
    downstream modules should never crash on missing fields.
    """
    source_path: Optional[str] = None       # original file path / S3 key
    title: Optional[str] = None             # extracted document title
    author: Optional[str] = None
    created_date: Optional[str] = None      # ISO string from doc metadata
    modified_date: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    department: Optional[str] = None        # HR, Engineering, Legal …
    doc_category: Optional[str] = None      # SOP, policy, manual, notes …
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)   # anything else


class DocumentSection(BaseModel):
    """
    A logical section inside a document (heading + text block).
    Parsers that detect structure produce a list of these.
    """
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    heading: Optional[str] = None
    level: int = 0                          # heading depth (0 = body, 1 = H1 …)
    text: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    has_table: bool = False
    has_image: bool = False


class Document(BaseModel):
    """
    The canonical object produced by any Parser.
    Passed directly into the Chunker.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    doc_type: DocumentType = DocumentType.UNKNOWN
    raw_text: str = ""                      # full plain text (all pages joined)
    sections: List[DocumentSection] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_sections(self) -> bool:
        return len(self.sections) > 0

    @property
    def text_for_chunking(self) -> str:
        """
        Return the best text source for the chunker.
        Prefer section-structured text if available.
        """
        if self.has_sections:
            return "\n\n".join(
                f"{'#' * s.level} {s.heading}\n{s.text}".strip()
                if s.heading else s.text
                for s in self.sections
            )
        return self.raw_text


# ─────────────────────────── Chunk ───────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata carried with every chunk, surfaced in search results."""
    doc_id: str
    doc_filename: str
    doc_type: DocumentType = DocumentType.UNKNOWN
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    section_id: Optional[str] = None
    chunk_index: int = 0                    # position in the document
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    token_count: Optional[int] = None
    char_count: int = 0
    # searchable filter fields (inherited from DocumentMetadata)
    department: Optional[str] = None
    doc_category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """
    A text chunk produced by a Chunker.
    id is used as the primary key in the vector store.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None   # populated after embedding step

    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id

    @property
    def char_count(self) -> int:
        return len(self.text)


class EmbeddedChunk(Chunk):
    """A Chunk whose embedding field is guaranteed to be populated."""
    embedding: List[float]


# ─────────────────────────── Search / Retrieval ───────────────────────────────

class SearchResult(BaseModel):
    """Raw result returned by a VectorStore search."""
    chunk_id: str
    text: str
    score: float                            # higher = more relevant (normalised 0-1)
    metadata: ChunkMetadata
    search_type: str = "dense"              # dense | keyword | hybrid


class RetrievedChunk(BaseModel):
    """
    A search result that has been re-ranked and is ready for
    the Context Builder. Carries the original score + re-rank score.
    """
    chunk_id: str
    text: str
    score: float                            # final score after re-ranking
    original_score: float                   # score from the raw search step
    rerank_score: Optional[float] = None    # score from the re-ranker (if used)
    metadata: ChunkMetadata
    rank: int = 0                           # 1-based position in the final list


class QueryContext(BaseModel):
    """
    The assembled context sent to the LLM, built by the ContextBuilder.
    Carries all chunks used + token accounting.
    """
    query: str
    chunks: List[RetrievedChunk]
    formatted_context: str                  # the actual text block injected into prompt
    total_tokens: int = 0
    truncated: bool = False                 # True if some chunks were cut for token budget


# ─────────────────────────── Answer / Citation ───────────────────────────────

class Citation(BaseModel):
    """A single source citation attached to an answer."""
    citation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str
    doc_id: str
    doc_filename: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    excerpt: str = ""                       # the specific passage cited


class Answer(BaseModel):
    """
    The final answer object returned to the user.
    Combines LLM-generated text with traceable citations.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer_text: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[float] = None      # 0-1, set by the evaluation layer
    model_used: str = ""
    retrieval_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────── Job / Feedback ──────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class Job(BaseModel):
    """Represents an async ingestion task."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    job_type: str = "ingest"
    status: JobStatus = JobStatus.PENDING
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Feedback(BaseModel):
    """User feedback on an answer — used for RLHF signal and re-ranking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer_id: str
    query: str
    answer_text: str
    rating: int = Field(ge=1, le=5)         # 1=terrible … 5=perfect
    correction: Optional[str] = None        # user's corrected answer
    bad_citations: List[str] = Field(default_factory=list)  # chunk_ids that were wrong
    created_at: datetime = Field(default_factory=datetime.utcnow)
