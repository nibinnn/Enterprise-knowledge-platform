"""
tests/unit/test_rag_and_agents.py
─────────────────────────────────────────────────────────────────────────────
Tests for: context_builder, citation engine, reranker, retriever,
           RAG pipeline, and all four agent tools.
All external dependencies (LLM, Qdrant, embedder) are mocked.
Run with:  pytest tests/unit/test_rag_and_agents.py -v
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import math
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.models.document import (
    Answer, ChunkMetadata, Citation, DocumentType,
    QueryContext, RetrievedChunk, SearchResult,
)


# ─────────────────────────── helpers ─────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_chunk(text: str, doc_id: str = "doc-1", score: float = 0.9, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        score=score,
        original_score=score,
        rank=rank,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            doc_filename="test.pdf",
            doc_type=DocumentType.PDF,
            chunk_index=0,
            page_number=1,
            section_heading="Introduction",
        ),
    )


def make_search_result(text: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        text=text,
        score=score,
        metadata=ChunkMetadata(
            doc_id="doc-1",
            doc_filename="test.pdf",
            doc_type=DocumentType.PDF,
            chunk_index=0,
        ),
        search_type="dense",
    )


# ─────────────────────────── ContextBuilder ──────────────────────────────────

class TestContextBuilder:

    def _builder(self, max_tokens=1000):
        from app.rag.context_builder import ContextBuilder
        return ContextBuilder(max_tokens=max_tokens)

    def test_empty_chunks_returns_no_sources_message(self):
        ctx = self._builder().build("What is X?", [])
        assert "No relevant sources" in ctx.formatted_context

    def test_single_chunk_included(self):
        chunk = make_chunk("The policy states employees get 20 days of leave.")
        ctx   = self._builder().build("What is the leave policy?", [chunk])
        assert "20 days" in ctx.formatted_context
        assert "Source [1]" in ctx.formatted_context

    def test_source_header_includes_filename(self):
        chunk = make_chunk("Some content.")
        ctx   = self._builder().build("query", [chunk])
        assert "test.pdf" in ctx.formatted_context

    def test_source_header_includes_page(self):
        chunk = make_chunk("Content here.")
        ctx   = self._builder().build("query", [chunk])
        assert "p.1" in ctx.formatted_context

    def test_source_header_includes_section(self):
        chunk = make_chunk("Content here.")
        ctx   = self._builder().build("query", [chunk])
        assert "Introduction" in ctx.formatted_context

    def test_token_budget_truncates(self):
        # Very small budget forces truncation
        chunks = [make_chunk("Word " * 200) for _ in range(5)]
        ctx    = self._builder(max_tokens=50).build("query", chunks)
        assert ctx.truncated is True
        assert len(ctx.chunks) < 5

    def test_deduplication_removes_near_identical(self):
        text = "This is the exact same content appearing twice."
        c1   = make_chunk(text, doc_id="doc-1")
        c2   = make_chunk(text, doc_id="doc-1")
        ctx  = self._builder().build("query", [c1, c2])
        assert len(ctx.chunks) == 1

    def test_deduplication_keeps_different_docs(self):
        text = "Same text but different documents."
        c1   = make_chunk(text, doc_id="doc-1")
        c2   = make_chunk(text, doc_id="doc-2")
        ctx  = self._builder().build("query", [c1, c2])
        assert len(ctx.chunks) == 2

    def test_chunks_numbered_from_one(self):
        chunks = [make_chunk(f"Chunk {i}") for i in range(3)]
        ctx    = self._builder().build("query", chunks)
        assert "Source [1]" in ctx.formatted_context
        assert "Source [2]" in ctx.formatted_context
        assert "Source [3]" in ctx.formatted_context

    def test_query_context_has_correct_query(self):
        ctx = self._builder().build("What is AI?", [make_chunk("AI is...")])
        assert ctx.query == "What is AI?"

    def test_total_tokens_positive(self):
        ctx = self._builder().build("query", [make_chunk("Some text.")])
        assert ctx.total_tokens > 0


# ─────────────────────────── CitationEngine ──────────────────────────────────

class TestCitationEngine:

    def _engine(self):
        from app.citations.engine import CitationEngine
        return CitationEngine()

    def test_extracts_citation_numbers(self):
        engine = self._engine()
        chunks = [make_chunk("The refund policy allows 30 days.")]
        text   = "According to policy [1], refunds are available."
        _, citations = engine.extract_and_verify(text, chunks)
        assert len(citations) == 1
        assert citations[0].doc_filename == "test.pdf"

    def test_out_of_range_citation_skipped(self):
        engine = self._engine()
        chunks = [make_chunk("Only one chunk.")]
        text   = "See [5] for details."    # no chunk 5
        _, citations = engine.extract_and_verify(text, chunks)
        assert len(citations) == 0

    def test_multiple_citations_extracted(self):
        engine = self._engine()
        chunks = [make_chunk(f"Chunk {i}") for i in range(3)]
        text   = "Point A [1] and point B [2] and point C [3]."
        _, citations = engine.extract_and_verify(text, chunks)
        assert len(citations) == 3

    def test_duplicate_citation_not_repeated(self):
        engine = self._engine()
        chunks = [make_chunk("Some important info.")]
        text   = "As noted [1], this is important [1]."
        _, citations = engine.extract_and_verify(text, chunks)
        assert len(citations) == 1

    def test_answer_text_returned_unchanged(self):
        engine = self._engine()
        chunks = [make_chunk("Details here.")]
        original = "The answer is X [1]."
        returned_text, _ = engine.extract_and_verify(original, chunks)
        assert returned_text == original

    def test_excerpt_truncated_to_400_chars(self):
        engine = self._engine()
        long_text = "Word " * 200
        chunks    = [make_chunk(long_text)]
        _, citations = engine.extract_and_verify("See [1].", chunks)
        assert len(citations[0].excerpt) <= 410    # 400 + ellipsis

    def test_support_score_high_for_good_match(self):
        engine = self._engine()
        claim  = "The refund policy allows thirty days."
        source = "The company refund policy allows thirty days for returns."
        score  = engine._compute_support(claim, source)
        assert score > 0.5

    def test_support_score_low_for_poor_match(self):
        engine = self._engine()
        claim  = "Elephants migrate across the savanna."
        source = "Quarterly revenue increased by fifteen percent this year."
        score  = engine._compute_support(claim, source)
        assert score < 0.2

    def test_format_bibliography_empty(self):
        engine = self._engine()
        assert engine.format_bibliography([]) == ""

    def test_format_bibliography_non_empty(self):
        engine = self._engine()
        c = Citation(
            chunk_id="c1", doc_id="d1", doc_filename="policy.pdf",
            page_number=5, excerpt="excerpt",
        )
        bib = engine.format_bibliography([c])
        assert "policy.pdf" in bib
        assert "Sources" in bib


# ─────────────────────────── Reranker ────────────────────────────────────────

class TestNoOpReranker:

    def test_returns_top_n(self):
        from app.retrieval.reranker import NoOpReranker
        chunks  = [make_chunk(f"chunk {i}", rank=i+1) for i in range(10)]
        result  = run(NoOpReranker().rerank("query", chunks, top_n=3))
        assert len(result) == 3

    def test_rank_set_sequentially(self):
        from app.retrieval.reranker import NoOpReranker
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        result = run(NoOpReranker().rerank("query", chunks, top_n=5))
        for i, c in enumerate(result, 1):
            assert c.rank == i

    def test_empty_input_returns_empty(self):
        from app.retrieval.reranker import NoOpReranker
        assert run(NoOpReranker().rerank("query", [], top_n=5)) == []

    def test_top_n_larger_than_chunks(self):
        from app.retrieval.reranker import NoOpReranker
        chunks = [make_chunk("c1"), make_chunk("c2")]
        result = run(NoOpReranker().rerank("query", chunks, top_n=10))
        assert len(result) == 2


class TestGetReranker:

    def test_factory_returns_noop(self):
        from app.retrieval.reranker import NoOpReranker, get_reranker
        assert isinstance(get_reranker("none"), NoOpReranker)

    def test_factory_returns_cross_encoder(self):
        from app.retrieval.reranker import CrossEncoderReranker, get_reranker
        assert isinstance(get_reranker("cross_encoder"), CrossEncoderReranker)

    def test_factory_default_is_cross_encoder(self):
        from app.retrieval.reranker import CrossEncoderReranker, get_reranker
        assert isinstance(get_reranker(), CrossEncoderReranker)


# ─────────────────────────── HybridRetriever ─────────────────────────────────

class TestHybridRetriever:

    def _retriever(self, mode="hybrid"):
        from app.retrieval.retriever import HybridRetriever
        from app.retrieval.reranker  import NoOpReranker

        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])

        mock_vs = AsyncMock()
        mock_vs.search         = AsyncMock(return_value=[make_search_result("Dense result")])
        mock_vs.keyword_search = AsyncMock(return_value=[make_search_result("Keyword result")])
        mock_vs.hybrid_search  = AsyncMock(return_value=[make_search_result("Hybrid result")])

        return HybridRetriever(
            embedder=mock_embedder,
            vector_store=mock_vs,
            reranker=NoOpReranker(),
            search_mode=mode,
        ), mock_embedder, mock_vs

    def test_dense_mode_calls_search(self):
        retriever, _, mock_vs = self._retriever(mode="dense")
        chunks = run(retriever.retrieve("test query", top_k=5))
        mock_vs.search.assert_called_once()
        assert len(chunks) > 0

    def test_keyword_mode_calls_keyword_search(self):
        retriever, _, mock_vs = self._retriever(mode="keyword")
        run(retriever.retrieve("test query", top_k=5))
        mock_vs.keyword_search.assert_called_once()

    def test_hybrid_mode_calls_hybrid_search(self):
        retriever, _, mock_vs = self._retriever(mode="hybrid")
        run(retriever.retrieve("test query", top_k=5))
        mock_vs.hybrid_search.assert_called_once()

    def test_empty_query_returns_empty(self):
        retriever, _, _ = self._retriever()
        chunks = run(retriever.retrieve("   ", top_k=5))
        assert chunks == []

    def test_query_embedding_called(self):
        retriever, mock_embedder, _ = self._retriever()
        run(retriever.retrieve("test query", top_k=5))
        mock_embedder.embed_query.assert_called_once()

    def test_filters_passed_to_vector_store(self):
        retriever, _, mock_vs = self._retriever(mode="dense")
        run(retriever.retrieve("query", top_k=5, filters={"department": "HR"}))
        call_kwargs = mock_vs.search.call_args
        assert "HR" in str(call_kwargs)

    def test_rewrite_query_strips_whitespace(self):
        retriever, _, _ = self._retriever()
        rewritten = run(retriever.rewrite_query("  hello   world  "))
        assert rewritten == "hello world"

    def test_returned_chunks_have_rank(self):
        retriever, _, _ = self._retriever()
        chunks = run(retriever.retrieve("test query", top_k=5))
        for i, c in enumerate(chunks, 1):
            assert c.rank == i


# ─────────────────────────── RAG Pipeline ────────────────────────────────────

class TestRAGPipeline:

    def _pipeline(self, answer_text="The policy states 30 days [1]."):
        from app.rag.pipeline    import RAGPipeline
        from app.rag.llm         import LLMClient
        from app.rag.context_builder import ContextBuilder
        from app.retrieval.reranker  import NoOpReranker

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=[
            make_chunk("The refund policy allows 30 days.", rank=1),
            make_chunk("Employees must submit a form.", rank=2),
        ])

        mock_llm = AsyncMock(spec=LLMClient)
        mock_llm._model = "claude-sonnet-4-6"
        mock_llm.generate_answer = AsyncMock(return_value=answer_text)

        return RAGPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
            top_k=5,
            rerank_top_n=3,
        ), mock_retriever, mock_llm

    def test_run_returns_answer(self):
        pipeline, _, _ = self._pipeline()
        answer = run(pipeline.run("What is the refund policy?"))
        assert isinstance(answer, Answer)
        assert answer.answer_text == "The policy states 30 days [1]."

    def test_retriever_called_with_question(self):
        pipeline, mock_ret, _ = self._pipeline()
        run(pipeline.run("What is the refund policy?"))
        mock_ret.retrieve.assert_called_once()
        call_args = mock_ret.retrieve.call_args
        assert "refund policy" in call_args[1].get("query", call_args[0][0] if call_args[0] else "")

    def test_llm_called_with_context(self):
        pipeline, _, mock_llm = self._pipeline()
        run(pipeline.run("What is the refund policy?"))
        mock_llm.generate_answer.assert_called_once()

    def test_citations_extracted_from_answer(self):
        pipeline, _, _ = self._pipeline(answer_text="According to [1], refunds are 30 days.")
        answer = run(pipeline.run("Refund policy?"))
        assert len(answer.citations) == 1

    def test_no_citations_when_no_markers(self):
        pipeline, _, _ = self._pipeline(answer_text="The answer is 42.")
        answer = run(pipeline.run("What is 42?"))
        assert len(answer.citations) == 0

    def test_answer_has_latency_fields(self):
        pipeline, _, _ = self._pipeline()
        answer = run(pipeline.run("query"))
        assert answer.retrieval_latency_ms is not None
        assert answer.llm_latency_ms is not None
        assert answer.total_latency_ms is not None

    def test_answer_has_model_used(self):
        pipeline, _, _ = self._pipeline()
        answer = run(pipeline.run("query"))
        assert answer.model_used == "claude-sonnet-4-6"

    def test_filters_passed_to_retriever(self):
        pipeline, mock_ret, _ = self._pipeline()
        run(pipeline.run("query", filters={"department": "HR"}))
        call_kwargs = mock_ret.retrieve.call_args
        assert "HR" in str(call_kwargs)


# ─────────────────────────── Agent Tools ─────────────────────────────────────

class TestSearchTool:

    def _tool(self, chunks=None):
        from app.agents.tools.search_tool import SearchTool
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=chunks or [
            make_chunk("Policy allows 30 days refund.", rank=1),
        ])
        return SearchTool(retriever=mock_retriever)

    def test_name(self):
        from app.agents.tools.search_tool import SearchTool
        assert SearchTool.name == "search_knowledge_base"

    def test_has_parameters_schema(self):
        from app.agents.tools.search_tool import SearchTool
        schema = SearchTool.parameters_schema
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_returns_tool_result_ok(self):
        tool   = self._tool()
        result = run(tool.arun(query="refund policy"))
        assert result.success is True
        assert result.data is not None

    def test_results_in_data(self):
        tool   = self._tool()
        result = run(tool.arun(query="refund policy"))
        assert "results" in result.data
        assert len(result.data["results"]) > 0

    def test_empty_results(self):
        from app.agents.tools.search_tool import SearchTool
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=[])
        tool   = SearchTool(retriever=mock_retriever)
        result = run(tool.arun(query="nothing here"))
        assert result.success is True
        assert result.data["results"] == []

    def test_top_k_capped_at_20(self):
        from app.agents.tools.search_tool import SearchTool
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=[])
        tool = SearchTool(retriever=mock_retriever)
        run(tool.arun(query="test", top_k=999))
        call_kwargs = mock_retriever.retrieve.call_args
        assert call_kwargs[1].get("top_k", 999) <= 20

    def test_to_llm_schema(self):
        from app.agents.tools.search_tool import SearchTool
        schema = SearchTool(retriever=AsyncMock()).to_llm_schema()
        assert schema["name"] == "search_knowledge_base"
        assert "description" in schema
        assert "input_schema" in schema


class TestDocumentTool:

    def test_name(self):
        from app.agents.tools.document_tool import DocumentTool
        assert DocumentTool.name == "get_document"

    def test_requires_id_or_filename(self):
        from app.agents.tools.document_tool import DocumentTool
        tool   = DocumentTool()
        result = run(tool.arun())
        assert result.success is False

    def test_returns_ok_with_document_id(self):
        from app.agents.tools.document_tool import DocumentTool
        tool   = DocumentTool()
        result = run(tool.arun(document_id="doc-123"))
        assert result.success is True

    def test_returns_ok_with_filename(self):
        from app.agents.tools.document_tool import DocumentTool
        tool   = DocumentTool()
        result = run(tool.arun(filename="policy.pdf"))
        assert result.success is True


class TestSummarizationTool:

    def _tool(self, llm_response="A concise summary."):
        from app.agents.tools.summarization_tool import SummarizationTool
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=llm_response)
        return SummarizationTool(llm=mock_llm)

    def test_name(self):
        from app.agents.tools.summarization_tool import SummarizationTool
        assert SummarizationTool.name == "summarize"

    def test_returns_summary_in_output(self):
        tool   = self._tool("Brief summary here.")
        result = run(tool.arun(content="Long document content here."))
        assert result.success is True
        assert "Brief summary" in result.output

    def test_style_in_data(self):
        tool   = self._tool("Summary.")
        result = run(tool.arun(content="Content.", style="brief"))
        assert result.data["style"] == "brief"

    def test_focus_in_data(self):
        tool   = self._tool("Focused summary.")
        result = run(tool.arun(content="Content.", focus="cost savings"))
        assert result.data["focus"] == "cost savings"

    def test_llm_called_once(self):
        from app.agents.tools.summarization_tool import SummarizationTool
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="done")
        tool = SummarizationTool(llm=mock_llm)
        run(tool.arun(content="Some content to summarize."))
        mock_llm.complete.assert_called_once()


class TestAnalyticsTool:

    def _tool(self, chunks=None):
        from app.agents.tools.analytics_tool import AnalyticsTool
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=chunks or [
            make_chunk("Policy doc", doc_id="d1"),
            make_chunk("Another doc", doc_id="d2"),
        ])
        return AnalyticsTool(retriever=mock_retriever)

    def test_name(self):
        from app.agents.tools.analytics_tool import AnalyticsTool
        assert AnalyticsTool.name == "analyze_knowledge_base"

    def test_count_mode(self):
        tool   = self._tool()
        result = run(tool.arun(query="policy", analysis_type="count"))
        assert result.success is True
        assert result.data["matching_documents"] == 2

    def test_by_department_mode(self):
        tool   = self._tool()
        result = run(tool.arun(query="policy", analysis_type="by_department"))
        assert result.success is True
        assert "by_department" in result.data

    def test_by_doc_type_mode(self):
        tool   = self._tool()
        result = run(tool.arun(query="policy", analysis_type="by_doc_type"))
        assert result.success is True
        assert "by_doc_type" in result.data

    def test_unknown_analysis_type_fails(self):
        tool   = self._tool()
        result = run(tool.arun(query="test", analysis_type="invalid_type"))
        assert result.success is False


# ─────────────────────────── ToolRegistry ────────────────────────────────────

class TestToolRegistry:

    def test_register_and_get(self):
        from app.core.base.tool import ToolRegistry
        from app.agents.tools.search_tool import SearchTool
        registry = ToolRegistry()
        tool     = SearchTool(retriever=AsyncMock())
        registry.register(tool)
        assert registry.get("search_knowledge_base") is tool

    def test_get_unknown_raises(self):
        from app.core.base.tool import ToolRegistry
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_tool")

    def test_llm_schemas_all_tools(self):
        from app.core.base.tool import ToolRegistry
        from app.agents.tools.search_tool       import SearchTool
        from app.agents.tools.document_tool     import DocumentTool
        from app.agents.tools.summarization_tool import SummarizationTool
        from app.agents.tools.analytics_tool    import AnalyticsTool
        registry = ToolRegistry()
        for ToolClass in [SearchTool, DocumentTool, SummarizationTool, AnalyticsTool]:
            instance = ToolClass(retriever=AsyncMock()) if ToolClass in (SearchTool, AnalyticsTool) else ToolClass()
            registry.register(instance)
        schemas = registry.llm_schemas()
        assert len(schemas) == 4
        names = [s["name"] for s in schemas]
        assert "search_knowledge_base" in names
        assert "get_document"          in names
        assert "summarize"             in names
        assert "analyze_knowledge_base" in names

    def test_all_tools_count(self):
        from app.core.base.tool import ToolRegistry
        from app.agents.tools.search_tool       import SearchTool
        from app.agents.tools.document_tool     import DocumentTool
        from app.agents.tools.summarization_tool import SummarizationTool
        from app.agents.tools.analytics_tool    import AnalyticsTool
        registry = ToolRegistry()
        registry.register(SearchTool(retriever=AsyncMock()))
        registry.register(DocumentTool())
        registry.register(SummarizationTool())
        registry.register(AnalyticsTool(retriever=AsyncMock()))
        assert len(registry) == 4
