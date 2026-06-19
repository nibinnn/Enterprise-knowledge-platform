"""
tests/unit/test_models.py
─────────────────────────────────────────────────────────────────────────────
Day 1 unit tests — validate domain models, config, and base classes.
Run with:  pytest tests/unit/test_models.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
from app.core.models.document import (
    Answer, Chunk, ChunkMetadata, ChunkingStrategy, Citation,
    Document, DocumentMetadata, DocumentSection, DocumentStatus,
    DocumentType, EmbeddedChunk, Feedback, Job, JobStatus,
    QueryContext, RetrievedChunk, SearchResult,
)
from app.core.base.tool import ToolResult


# ─────────────────────────── Document ────────────────────────────────────────

class TestDocument:
    def test_default_id_is_uuid(self):
        doc = Document(filename="test.pdf")
        assert len(doc.id) == 36  # UUID format

    def test_two_documents_have_different_ids(self):
        d1 = Document(filename="a.pdf")
        d2 = Document(filename="b.pdf")
        assert d1.id != d2.id

    def test_text_for_chunking_uses_raw_text_when_no_sections(self):
        doc = Document(filename="test.pdf", raw_text="Hello world")
        assert doc.text_for_chunking == "Hello world"

    def test_text_for_chunking_uses_sections_when_available(self):
        sections = [
            DocumentSection(heading="Intro", level=1, text="This is intro."),
            DocumentSection(heading=None, level=0, text="Body text."),
        ]
        doc = Document(filename="test.pdf", raw_text="raw", sections=sections)
        chunking_text = doc.text_for_chunking
        assert "Intro" in chunking_text
        assert "This is intro." in chunking_text
        assert "Body text." in chunking_text

    def test_has_sections_property(self):
        doc = Document(filename="test.pdf")
        assert not doc.has_sections
        doc.sections.append(DocumentSection(text="section"))
        assert doc.has_sections

    def test_default_status_is_pending(self):
        doc = Document(filename="test.pdf")
        assert doc.status == DocumentStatus.PENDING

    def test_metadata_is_empty_by_default(self):
        doc = Document(filename="test.pdf")
        assert doc.metadata.tags == []
        assert doc.metadata.department is None


# ─────────────────────────── Chunk ───────────────────────────────────────────

class TestChunk:
    def _make_chunk(self, text="Sample text") -> Chunk:
        meta = ChunkMetadata(
            doc_id="doc-123",
            doc_filename="test.pdf",
            chunk_index=0,
        )
        return Chunk(text=text, metadata=meta)

    def test_chunk_has_unique_id(self):
        c1 = self._make_chunk()
        c2 = self._make_chunk()
        assert c1.id != c2.id

    def test_doc_id_property(self):
        chunk = self._make_chunk()
        assert chunk.doc_id == "doc-123"

    def test_char_count_property(self):
        chunk = self._make_chunk("hello")
        assert chunk.char_count == 5

    def test_embedding_is_none_by_default(self):
        chunk = self._make_chunk()
        assert chunk.embedding is None

    def test_embedded_chunk_requires_embedding(self):
        meta = ChunkMetadata(doc_id="x", doc_filename="x.pdf", chunk_index=0)
        with pytest.raises(Exception):  # missing required field
            EmbeddedChunk(text="text", metadata=meta)  # embedding missing

    def test_embedded_chunk_accepts_embedding(self):
        meta = ChunkMetadata(doc_id="x", doc_filename="x.pdf", chunk_index=0)
        ec = EmbeddedChunk(text="text", metadata=meta, embedding=[0.1, 0.2, 0.3])
        assert len(ec.embedding) == 3


# ─────────────────────────── SearchResult / RetrievedChunk ───────────────────

class TestRetrievedChunk:
    def _meta(self) -> ChunkMetadata:
        return ChunkMetadata(doc_id="d1", doc_filename="f.pdf", chunk_index=0)

    def test_retrieved_chunk_rank(self):
        rc = RetrievedChunk(
            chunk_id="c1", text="text", score=0.9, original_score=0.9,
            metadata=self._meta(), rank=1,
        )
        assert rc.rank == 1

    def test_rerank_score_optional(self):
        rc = RetrievedChunk(
            chunk_id="c1", text="text", score=0.9, original_score=0.9,
            metadata=self._meta(),
        )
        assert rc.rerank_score is None


# ─────────────────────────── QueryContext ────────────────────────────────────

class TestQueryContext:
    def test_context_not_truncated_by_default(self):
        ctx = QueryContext(query="q", chunks=[], formatted_context="ctx")
        assert not ctx.truncated

    def test_stores_query(self):
        ctx = QueryContext(query="What is X?", chunks=[], formatted_context="")
        assert ctx.query == "What is X?"


# ─────────────────────────── Answer / Citation ───────────────────────────────

class TestAnswer:
    def test_answer_has_id(self):
        a = Answer(query="q", answer_text="a")
        assert a.id

    def test_citations_empty_by_default(self):
        a = Answer(query="q", answer_text="a")
        assert a.citations == []

    def test_citation_structure(self):
        c = Citation(
            chunk_id="c1", doc_id="d1",
            doc_filename="doc.pdf", excerpt="excerpt text",
        )
        assert c.citation_id
        assert c.doc_filename == "doc.pdf"


# ─────────────────────────── Feedback ────────────────────────────────────────

class TestFeedback:
    def test_rating_range(self):
        # valid
        Feedback(answer_id="a1", query="q", answer_text="a", rating=5)
        # pydantic validation
        with pytest.raises(Exception):
            Feedback(answer_id="a1", query="q", answer_text="a", rating=6)
        with pytest.raises(Exception):
            Feedback(answer_id="a1", query="q", answer_text="a", rating=0)


# ─────────────────────────── ToolResult ──────────────────────────────────────

class TestToolResult:
    def test_ok_factory(self):
        r = ToolResult.ok("my_tool", "everything worked", {"key": "val"})
        assert r.success is True
        assert r.tool_name == "my_tool"
        assert r.data == {"key": "val"}

    def test_fail_factory(self):
        r = ToolResult.fail("my_tool", "something went wrong")
        assert r.success is False
        assert r.error == "something went wrong"
        assert "Tool error" in r.output


# ─────────────────────────── Config ──────────────────────────────────────────

class TestConfig:
    def test_settings_are_cached(self):
        from app.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2   # same object, lru_cache working

    def test_allowed_extension_list(self):
        from app.config import get_settings
        s = get_settings()
        exts = s.allowed_extension_list
        assert "pdf" in exts
        assert "docx" in exts

    def test_max_upload_size_bytes(self):
        from app.config import get_settings
        s = get_settings()
        assert s.max_upload_size_bytes == s.max_upload_size_mb * 1024 * 1024

    def test_hybrid_weight_validation(self):
        from pydantic import ValidationError
        from app.config import Settings
        with pytest.raises(ValidationError):
            Settings(hybrid_dense_weight=1.5)
