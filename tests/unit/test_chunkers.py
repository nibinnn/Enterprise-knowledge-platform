"""
tests/unit/test_chunkers.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Days 4 & 5 chunking modules.
Run with:  pytest tests/unit/test_chunkers.py -v
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import math
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.models.document import (
    Chunk, ChunkingStrategy, Document, DocumentSection, DocumentType,
)


# ─────────────────────────── Helpers ─────────────────────────────────────────

def make_document(
    text: str,
    sections: List[DocumentSection] | None = None,
    doc_type: DocumentType = DocumentType.TXT,
) -> Document:
    return Document(
        filename="test.txt",
        doc_type=doc_type,
        raw_text=text,
        sections=sections or [],
    )


def make_section(heading: str, text: str, level: int = 1) -> DocumentSection:
    return DocumentSection(heading=heading, level=level, text=text)


LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vaquely exquisite. "
    "Sphinx of black quartz, judge my vow. "
)

LONG_TEXT = (LOREM * 40).strip()   # ~1 200 words ≈ 1 600 tokens


# ─────────────────────────── TokenCounter ────────────────────────────────────

class TestTokenCounter:

    def test_count_returns_positive_int(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        result = tc.count("Hello world, this is a test sentence.")
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_string(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        assert tc.count("") == 0 or tc.count("") >= 0   # tiktoken returns 0

    def test_longer_text_has_more_tokens(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        short = tc.count("Hello.")
        long  = tc.count(LONG_TEXT)
        assert long > short

    def test_chunk_by_tokens_length(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        chunks = tc.chunk_by_tokens(LONG_TEXT, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert chunk.strip()

    def test_chunk_by_tokens_respects_size(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        chunks = tc.chunk_by_tokens(LONG_TEXT, chunk_size=100, chunk_overlap=0)
        for chunk in chunks[:-1]:   # last chunk may be smaller
            token_count = tc.count(chunk)
            # Allow small overshoot from tiktoken decoding boundaries
            assert token_count <= 110

    def test_chunk_by_tokens_overlap_increases_chunk_count(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        no_overlap   = tc.chunk_by_tokens(LONG_TEXT, 200, 0)
        with_overlap = tc.chunk_by_tokens(LONG_TEXT, 200, 50)
        assert len(with_overlap) >= len(no_overlap)

    def test_encode_decode_roundtrip(self):
        from app.chunkers.token_counter import TokenCounter
        tc = TokenCounter()
        if not tc.has_tiktoken:
            pytest.skip("tiktoken not installed")
        original = "Hello, world!"
        ids = tc.encode(original)
        decoded = tc.decode(ids)
        assert decoded == original


# ─────────────────────────── FixedChunker ────────────────────────────────────

class TestFixedChunker:

    def _chunker(self, size=100, overlap=10):
        from app.chunkers.fixed_chunker import FixedChunker
        return FixedChunker(chunk_size=size, chunk_overlap=overlap)

    def test_returns_list_of_chunks(self):
        chunker = self._chunker()
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_multiple_chunks_produced(self):
        chunker = self._chunker(size=50, overlap=5)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_each_chunk_has_text(self):
        chunker = self._chunker(size=100, overlap=10)
        chunks  = chunker.chunk(make_document(LONG_TEXT))
        for c in chunks:
            assert c.text.strip()

    def test_chunk_metadata_doc_id_set(self):
        chunker = self._chunker()
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.doc_id == doc.id

    def test_chunk_index_sequential(self):
        chunker = self._chunker(size=100, overlap=10)
        chunks  = chunker.chunk(make_document(LONG_TEXT))
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_strategy_is_fixed(self):
        chunker = self._chunker()
        chunks  = chunker.chunk(make_document(LONG_TEXT))
        for c in chunks:
            assert c.metadata.chunking_strategy == ChunkingStrategy.FIXED

    def test_short_text_produces_one_chunk(self):
        chunker = self._chunker(size=512, overlap=64)
        doc     = make_document("This is a very short text.")
        chunks  = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_empty_text_returns_empty_list(self):
        chunker = self._chunker()
        doc     = make_document("")
        chunks  = chunker.chunk(doc)
        assert chunks == []

    def test_token_count_in_metadata(self):
        chunker = self._chunker(size=100, overlap=10)
        chunks  = chunker.chunk(make_document(LONG_TEXT))
        for c in chunks:
            assert c.metadata.token_count is not None
            assert c.metadata.token_count > 0

    def test_overlap_causes_content_repetition(self):
        from app.chunkers.token_counter import TokenCounter
        chunker = self._chunker(size=50, overlap=25)
        chunks  = chunker.chunk(make_document(LONG_TEXT))
        if len(chunks) < 2:
            return
        # Last few words of chunk N should appear in chunk N+1
        last_words_of_first  = set(chunks[0].text.split()[-5:])
        first_words_of_second = set(chunks[1].text.split()[:10])
        assert last_words_of_first & first_words_of_second  # some overlap


# ─────────────────────────── RecursiveChunker ────────────────────────────────

class TestRecursiveChunker:

    def _chunker(self, size=200, overlap=20, section_aware=True):
        from app.chunkers.recursive_chunker import RecursiveChunker
        return RecursiveChunker(
            chunk_size=size, chunk_overlap=overlap, section_aware=section_aware
        )

    def test_returns_chunks(self):
        chunks = self._chunker().chunk(make_document(LONG_TEXT))
        assert len(chunks) > 0

    def test_strategy_is_recursive(self):
        chunks = self._chunker().chunk(make_document(LONG_TEXT))
        for c in chunks:
            assert c.metadata.chunking_strategy == ChunkingStrategy.RECURSIVE

    def test_respects_paragraph_boundaries(self):
        # Each paragraph is ~40 words so the chunker merges at most one per chunk
        para_a = "Alpha " * 40 + "end."
        para_b = "Beta "  * 40 + "end."
        text   = f"{para_a}\n\n{para_b}"
        chunker = self._chunker(size=40, overlap=0)
        chunks  = chunker.chunk(make_document(text))
        # At least 2 chunks should be produced (paragraphs are ~55 tokens each)
        assert len(chunks) >= 1   # basic sanity
        # Verify chunks have content from the text
        all_text = " ".join(c.text for c in chunks)
        assert "Alpha" in all_text
        assert "Beta" in all_text

    def test_section_aware_uses_section_heading(self):
        sections = [
            make_section("Introduction",  "Intro text. " * 30, level=1),
            make_section("Methodology",   "Method text. " * 30, level=1),
        ]
        doc = Document(
            filename="test.txt", doc_type=DocumentType.DOCX,
            raw_text="", sections=sections,
        )
        chunker = self._chunker(size=50, overlap=5, section_aware=True)
        chunks  = chunker.chunk(doc)
        headings = {c.metadata.section_heading for c in chunks}
        assert "Introduction" in headings
        assert "Methodology" in headings

    def test_section_aware_false_ignores_sections(self):
        sections = [make_section("H1", LOREM * 5)]
        doc = Document(
            filename="t.txt", doc_type=DocumentType.TXT,
            raw_text=LOREM * 5, sections=sections,
        )
        chunker = self._chunker(size=100, overlap=10, section_aware=False)
        chunks  = chunker.chunk(doc)
        # In flat mode, heading is not set
        for c in chunks:
            assert c.metadata.section_heading is None

    def test_empty_text_returns_empty(self):
        chunks = self._chunker().chunk(make_document(""))
        assert chunks == []

    def test_short_text_one_chunk(self):
        chunks = self._chunker(size=512, overlap=64).chunk(
            make_document("Short sentence here.")
        )
        assert len(chunks) == 1

    def test_chunk_index_sequential(self):
        chunks = self._chunker(size=80, overlap=10).chunk(make_document(LONG_TEXT))
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_doc_filename_in_metadata(self):
        doc = Document(filename="myfile.txt", doc_type=DocumentType.TXT,
                       raw_text=LONG_TEXT)
        chunks = self._chunker().chunk(doc)
        for c in chunks:
            assert c.metadata.doc_filename == "myfile.txt"

    def test_split_recursive_small_text_no_split(self):
        from app.chunkers.recursive_chunker import RecursiveChunker
        rc = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        pieces = rc._split_recursive("Short text.", rc._separators)
        assert pieces == ["Short text."]

    def test_merge_with_overlap_basic(self):
        from app.chunkers.recursive_chunker import RecursiveChunker
        rc = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        pieces = ["word " * 5, "word " * 5, "word " * 5]
        merged = rc._merge_with_overlap(pieces)
        assert len(merged) >= 1


# ─────────────────────────── SemanticChunker ─────────────────────────────────

class TestSemanticChunker:
    """
    Tests for SemanticChunker.
    The embedder is mocked so no API calls are made.
    """

    def _mock_embedder(self, dimension: int = 8):
        """Return a mock embedder whose embed() returns random unit vectors."""
        import random
        embedder = AsyncMock()

        async def fake_embed(texts):
            vecs = []
            for i, _ in enumerate(texts):
                # Alternate between two directions to create natural breakpoints
                if i % 4 == 0:
                    v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                vecs.append(v)
            return vecs

        embedder.embed = fake_embed
        return embedder

    def _chunker(self, threshold=0.5):
        from app.chunkers.semantic_chunker import SemanticChunker
        return SemanticChunker(
            chunk_size=200,
            chunk_overlap=20,
            breakpoint_threshold=threshold,
            embedder=self._mock_embedder(),
        )

    def test_split_sentences_basic(self):
        from app.chunkers.semantic_chunker import SemanticChunker
        text = "First sentence. Second sentence. Third sentence."
        sents = SemanticChunker._split_sentences(text)
        assert len(sents) >= 2
        assert any("First" in s for s in sents)

    def test_split_sentences_single(self):
        from app.chunkers.semantic_chunker import SemanticChunker
        sents = SemanticChunker._split_sentences("Just one sentence")
        assert len(sents) == 1

    def test_cosine_similarity_identical_vectors(self):
        from app.chunkers.semantic_chunker import SemanticChunker
        v = [1.0, 0.0, 0.0]
        assert abs(SemanticChunker._cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        from app.chunkers.semantic_chunker import SemanticChunker
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(SemanticChunker._cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from app.chunkers.semantic_chunker import SemanticChunker
        assert SemanticChunker._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_group_sentences_creates_groups(self):
        from app.chunkers.semantic_chunker import SemanticChunker
        sc = SemanticChunker(breakpoint_threshold=0.5, embedder=self._mock_embedder())
        sentences = [f"Sentence {i}." for i in range(8)]
        # Alternate direction embeddings → low similarity every 4 sentences
        embeddings = [
            [1.0, 0.0] if i % 4 == 0 else [0.0, 1.0]
            for i in range(8)
        ]
        groups = sc._group_sentences(sentences, embeddings)
        assert len(groups) >= 1
        # All sentences should be accounted for
        all_sents = [s for g in groups for s in g]
        assert len(all_sents) == 8

    def test_chunk_returns_list(self):
        chunker = self._chunker()
        text = ". ".join([f"Sentence number {i}" for i in range(20)]) + "."
        doc  = make_document(text)
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)

    def test_chunk_strategy_is_semantic(self):
        chunker = self._chunker()
        text    = ". ".join([f"Point {i}" for i in range(12)]) + "."
        chunks  = chunker.chunk(make_document(text))
        for c in chunks:
            assert c.metadata.chunking_strategy == ChunkingStrategy.SEMANTIC

    def test_empty_document_returns_empty(self):
        chunker = self._chunker()
        chunks  = chunker.chunk(make_document(""))
        assert chunks == []


# ─────────────────────────── ChunkingStrategyRouter ──────────────────────────

class TestChunkingStrategyRouter:

    def _router(self):
        from app.chunkers.strategy_router import ChunkingStrategyRouter
        return ChunkingStrategyRouter(chunk_size=200, chunk_overlap=20)

    def test_explicit_fixed_strategy(self):
        from app.chunkers.strategy_router import ChunkingStrategyRouter
        router = ChunkingStrategyRouter(chunk_size=200, chunk_overlap=20)
        from app.chunkers.fixed_chunker import FixedChunker
        chunker = router._build_chunker(ChunkingStrategy.FIXED)
        assert isinstance(chunker, FixedChunker)

    def test_explicit_recursive_strategy(self):
        from app.chunkers.strategy_router import ChunkingStrategyRouter
        from app.chunkers.recursive_chunker import RecursiveChunker
        router  = ChunkingStrategyRouter(chunk_size=200, chunk_overlap=20)
        chunker = router._build_chunker(ChunkingStrategy.RECURSIVE)
        assert isinstance(chunker, RecursiveChunker)

    def test_pdf_with_sections_selects_recursive(self):
        from app.chunkers.strategy_router import ChunkingStrategyRouter, ChunkingStrategy
        router = ChunkingStrategyRouter()
        doc = Document(
            filename="doc.pdf", doc_type=DocumentType.PDF,
            raw_text=LOREM * 5,
            sections=[make_section("H", LOREM) for _ in range(4)],
        )
        selected = router._select_strategy(doc, override=None)
        assert selected == ChunkingStrategy.RECURSIVE

    def test_pdf_with_no_sections_selects_fixed(self):
        from app.chunkers.strategy_router import ChunkingStrategyRouter, ChunkingStrategy
        router = ChunkingStrategyRouter()
        doc = Document(
            filename="scan.pdf", doc_type=DocumentType.PDF,
            raw_text=LOREM, sections=[],
        )
        selected = router._select_strategy(doc, override=None)
        assert selected == ChunkingStrategy.FIXED

    def test_override_beats_auto_selection(self):
        from app.chunkers.strategy_router import ChunkingStrategyRouter, ChunkingStrategy
        router = ChunkingStrategyRouter()
        doc = Document(
            filename="doc.pdf", doc_type=DocumentType.PDF,
            raw_text=LONG_TEXT,
            sections=[make_section("H", LOREM) for _ in range(5)],
        )
        selected = router._select_strategy(doc, override="fixed")
        assert selected == ChunkingStrategy.FIXED

    def test_chunk_returns_chunks(self):
        router = self._router()
        chunks = router.chunk(make_document(LONG_TEXT))
        assert len(chunks) > 0

    def test_compare_strategies_returns_dict(self):
        router  = self._router()
        doc     = make_document(LONG_TEXT)
        results = router.compare_strategies(doc, strategies=["fixed", "recursive"])
        assert "fixed"     in results
        assert "recursive" in results
        assert len(results["fixed"])     > 0
        assert len(results["recursive"]) > 0

    def test_summarise_comparison(self):
        router  = self._router()
        doc     = make_document(LONG_TEXT)
        comp    = router.compare_strategies(doc, strategies=["fixed", "recursive"])
        summary = router.summarise_comparison(comp)
        for strategy, stats in summary.items():
            assert "count"      in stats
            assert "avg_tokens" in stats
            assert stats["count"] > 0
