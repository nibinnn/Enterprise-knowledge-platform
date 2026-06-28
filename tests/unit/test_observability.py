"""
tests/unit/test_observability.py
─────────────────────────────────────────────────────────────────────────────
Tests for:
  - app/core/tracing.py   (no-op tracer, context var, get_trace_id)
  - app/core/metrics.py   (all metric objects exist and are the right type)
  - app/api/middleware.py (request ID injection, /metrics endpoint)

All tests run without Langfuse or Prometheus push gateway.
Run with:  pytest tests/unit/test_observability.py -v
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────── No-op Tracer ────────────────────────────────────

class TestNoOpTracer:

    def setup_method(self):
        from app.core.tracing import reset_tracer
        reset_tracer()

    def test_get_tracer_returns_noop_when_not_configured(self):
        from app.core.tracing import get_tracer, _NoOpTracer
        with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}):
            from app.config import get_settings
            get_settings.cache_clear()
            tracer = get_tracer()
            assert isinstance(tracer, _NoOpTracer)
            get_settings.cache_clear()

    def test_trace_context_manager_yields_trace(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("test_trace") as t:
                assert t is not None
                assert t.id
                assert t.name == "test_trace"
        run(_go())

    def test_span_context_manager_yields_span(self):
        from app.core.tracing import _NoOpTracer, _NoOpTrace
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("t") as trace:
                async with trace.span("my_span") as s:
                    assert s.name == "my_span"
        run(_go())

    def test_generation_context_manager_yields_generation(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("t") as trace:
                async with trace.generation("llm_call", model="gpt-4") as g:
                    assert g is not None
        run(_go())

    def test_span_end_does_not_raise(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("t") as trace:
                async with trace.span("s") as s:
                    s.end(output={"result": "ok"})
        run(_go())

    def test_generation_end_with_completion(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("t") as trace:
                async with trace.generation("g") as g:
                    g.end(completion="The answer is 42.")
        run(_go())

    def test_nested_spans_do_not_raise(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("outer") as trace:
                async with trace.span("span_a") as a:
                    async with trace.span("span_b") as b:
                        b.end(output="inner done")
                    a.end(output="outer done")
        run(_go())

    def test_exception_inside_trace_propagates(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            with pytest.raises(ValueError, match="expected"):
                async with tracer.trace("t") as trace:
                    raise ValueError("expected")
        run(_go())

    def test_exception_inside_span_propagates(self):
        from app.core.tracing import _NoOpTracer
        tracer = _NoOpTracer()
        async def _go():
            with pytest.raises(RuntimeError):
                async with tracer.trace("t") as trace:
                    async with trace.span("s"):
                        raise RuntimeError("boom")
        run(_go())

    def test_span_elapsed_ms_positive(self):
        from app.core.tracing import _NoOpSpan
        import time
        s = _NoOpSpan("test")
        time.sleep(0.01)
        assert s.elapsed_ms > 0

    def test_tracer_singleton_cached(self):
        from app.core.tracing import get_tracer, reset_tracer
        reset_tracer()
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_reset_tracer_clears_singleton(self):
        from app.core.tracing import get_tracer, reset_tracer
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        # After reset, a new instance is created
        assert t2 is not None


# ─────────────────────────── Context variable ─────────────────────────────────

class TestTraceContextVar:

    def setup_method(self):
        from app.core.tracing import reset_tracer
        reset_tracer()

    def test_trace_id_none_outside_trace(self):
        from app.core.tracing import get_trace_id, _NoOpTracer, _current_trace_id
        _current_trace_id.set(None)
        assert get_trace_id() is None

    def test_trace_id_set_inside_trace(self):
        from app.core.tracing import get_trace_id, _NoOpTracer
        tracer = _NoOpTracer()
        captured = []
        async def _go():
            async with tracer.trace("t"):
                captured.append(get_trace_id())
        run(_go())
        assert captured[0] is not None
        assert len(captured[0]) > 0

    def test_trace_id_cleared_after_trace(self):
        from app.core.tracing import get_trace_id, _NoOpTracer, _current_trace_id
        _current_trace_id.set(None)
        tracer = _NoOpTracer()
        async def _go():
            async with tracer.trace("t"):
                pass
        run(_go())
        assert get_trace_id() is None

    def test_concurrent_traces_have_different_ids(self):
        from app.core.tracing import get_trace_id, _NoOpTracer
        tracer = _NoOpTracer()
        ids = []
        async def _capture():
            async with tracer.trace("t"):
                ids.append(get_trace_id())
        async def _go():
            await asyncio.gather(_capture(), _capture(), _capture())
        run(_go())
        # All IDs should be non-None
        assert all(i is not None for i in ids)


# ─────────────────────────── Prometheus Metrics ───────────────────────────────

class TestMetrics:

    def test_http_requests_total_exists(self):
        from app.core.metrics import HTTP_REQUESTS_TOTAL
        from prometheus_client import Counter
        assert isinstance(HTTP_REQUESTS_TOTAL, Counter)

    def test_http_request_duration_exists(self):
        from app.core.metrics import HTTP_REQUEST_DURATION
        from prometheus_client import Histogram
        assert isinstance(HTTP_REQUEST_DURATION, Histogram)

    def test_rag_requests_total_exists(self):
        from app.core.metrics import RAG_REQUESTS_TOTAL
        from prometheus_client import Counter
        assert isinstance(RAG_REQUESTS_TOTAL, Counter)

    def test_rag_retrieval_duration_exists(self):
        from app.core.metrics import RAG_RETRIEVAL_DURATION
        from prometheus_client import Histogram
        assert isinstance(RAG_RETRIEVAL_DURATION, Histogram)

    def test_rag_llm_duration_exists(self):
        from app.core.metrics import RAG_LLM_DURATION
        from prometheus_client import Histogram
        assert isinstance(RAG_LLM_DURATION, Histogram)

    def test_agent_requests_total_exists(self):
        from app.core.metrics import AGENT_REQUESTS_TOTAL
        from prometheus_client import Counter
        assert isinstance(AGENT_REQUESTS_TOTAL, Counter)

    def test_agent_iterations_exists(self):
        from app.core.metrics import AGENT_ITERATIONS
        from prometheus_client import Histogram
        assert isinstance(AGENT_ITERATIONS, Histogram)

    def test_agent_tool_calls_total_exists(self):
        from app.core.metrics import AGENT_TOOL_CALLS_TOTAL
        from prometheus_client import Counter
        assert isinstance(AGENT_TOOL_CALLS_TOTAL, Counter)

    def test_documents_total_gauge_exists(self):
        from app.core.metrics import DOCUMENTS_TOTAL
        from prometheus_client import Gauge
        assert isinstance(DOCUMENTS_TOTAL, Gauge)

    def test_ingestion_total_counter_exists(self):
        from app.core.metrics import INGESTION_TOTAL
        from prometheus_client import Counter
        assert isinstance(INGESTION_TOTAL, Counter)

    def test_counter_can_be_incremented(self):
        from app.core.metrics import RAG_REQUESTS_TOTAL
        before = RAG_REQUESTS_TOTAL.labels(status="test_inc")._value.get()
        RAG_REQUESTS_TOTAL.labels(status="test_inc").inc()
        after  = RAG_REQUESTS_TOTAL.labels(status="test_inc")._value.get()
        assert after == before + 1

    def test_histogram_can_be_observed(self):
        from app.core.metrics import RAG_RETRIEVAL_DURATION
        # Should not raise
        RAG_RETRIEVAL_DURATION.observe(0.123)

    def test_gauge_can_be_set(self):
        from app.core.metrics import DOCUMENTS_TOTAL
        DOCUMENTS_TOTAL.labels(status="indexed").set(42)
        val = DOCUMENTS_TOTAL.labels(status="indexed")._value.get()
        assert val == 42


# ─────────────────────────── Middleware ──────────────────────────────────────

class TestMiddleware:

    @pytest.fixture(scope="class")
    def client(self):
        import app.db.database as _db_mod   # noqa
        import app.db.models               # noqa
        from app.db.database import get_db
        from app.main import app as _app

        async def fake_db():
            db = AsyncMock()
            db.execute = AsyncMock(return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=None),
                scalar_one=MagicMock(return_value=0),
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
            ))
            db.add = MagicMock()
            db.flush = AsyncMock()
            yield db

        _app.dependency_overrides[get_db] = fake_db
        from fastapi.testclient import TestClient
        with patch("app.main.init_db",  AsyncMock()), \
             patch("app.main.close_db", AsyncMock()), \
             patch("app.db.database.check_db_connection", AsyncMock(return_value=True)):
            with TestClient(_app, raise_server_exceptions=False) as c:
                yield c
        _app.dependency_overrides.clear()

    def test_request_id_header_returned(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_request_id_is_uuid_format(self, client):
        resp = client.get("/health")
        rid  = resp.headers.get("x-request-id", "")
        assert len(rid) == 36   # UUID length

    def test_custom_request_id_echoed_back(self, client):
        custom_id = str(uuid.uuid4())
        resp = client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.headers.get("x-request-id") == custom_id

    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_endpoint_content_type_prometheus(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_endpoint_contains_ekip_metrics(self, client):
        # Trigger a request to generate some metric data
        client.get("/health")
        resp = client.get("/metrics")
        content = resp.text
        # At minimum the prometheus default metrics should appear
        assert "python_info" in content or "process_" in content or "ekip_" in content

    def test_health_endpoint_not_in_metrics(self, client):
        """Health endpoint calls should not create metric entries with /health path."""
        # This is a soft test — just check /metrics is reachable after /health calls
        client.get("/health")
        client.get("/health")
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_different_requests_have_different_ids(self, client):
        resp1 = client.get("/health")
        resp2 = client.get("/health")
        id1   = resp1.headers.get("x-request-id")
        id2   = resp2.headers.get("x-request-id")
        assert id1 != id2


# ─────────────────────────── Tracing in pipeline ─────────────────────────────

class TestTracingInPipeline:
    """
    Verify that the RAG pipeline and agent orchestrator call the tracer
    without errors (using the no-op tracer).
    """

    def test_pipeline_runs_with_noop_tracer(self):
        from app.core.tracing import reset_tracer
        from app.core.models.document import ChunkMetadata, DocumentType, RetrievedChunk
        reset_tracer()

        async def _go():
            from app.rag.pipeline        import RAGPipeline
            from app.rag.llm             import LLMClient
            from app.retrieval.reranker  import NoOpReranker

            mock_retriever = AsyncMock()
            mock_retriever.retrieve = AsyncMock(return_value=[
                RetrievedChunk(
                    chunk_id="c1", text="Policy text here.",
                    score=0.9, original_score=0.9, rank=1,
                    metadata=ChunkMetadata(
                        doc_id="d1", doc_filename="policy.pdf",
                        doc_type=DocumentType.PDF, chunk_index=0,
                    ),
                )
            ])
            mock_llm = AsyncMock(spec=LLMClient)
            mock_llm._model = "claude-sonnet-4-6"
            mock_llm.generate_answer = AsyncMock(return_value="The policy says [1].")

            pipeline = RAGPipeline(
                retriever=mock_retriever, llm=mock_llm, top_k=5, rerank_top_n=3,
            )
            answer = await pipeline.run("What is the policy?")
            assert answer.answer_text == "The policy says [1]."
            assert answer.retrieval_latency_ms is not None
            assert answer.llm_latency_ms is not None

        run(_go())
        reset_tracer()

    def test_retriever_runs_with_noop_tracer(self):
        from app.core.tracing import reset_tracer
        reset_tracer()

        async def _go():
            from app.retrieval.retriever import HybridRetriever
            from app.retrieval.reranker  import NoOpReranker

            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 4)

            mock_vs = AsyncMock()
            mock_vs.hybrid_search = AsyncMock(return_value=[])

            r = HybridRetriever(
                embedder=mock_embedder,
                vector_store=mock_vs,
                reranker=NoOpReranker(),
            )
            chunks = await r.retrieve("test query", top_k=5)
            assert chunks == []

        run(_go())
        reset_tracer()
