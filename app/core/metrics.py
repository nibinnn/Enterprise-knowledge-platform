"""
app/core/metrics.py
─────────────────────────────────────────────────────────────────────────────
Prometheus metrics for the EKIP platform.

Metrics exposed at GET /metrics (added to main.py):

  ekip_http_requests_total          counter   HTTP requests by method/path/status
  ekip_http_request_duration_s      histogram HTTP request latency by path
  ekip_rag_retrieval_duration_s     histogram Retrieval step latency
  ekip_rag_llm_duration_s           histogram LLM generation latency
  ekip_rag_requests_total           counter   Total RAG requests by status
  ekip_agent_requests_total         counter   Total agent requests by status
  ekip_agent_iterations             histogram Agent tool-call iterations per request
  ekip_documents_total              gauge     Documents by status (polled from DB)
  ekip_chunks_total                 gauge     Total indexed chunks
  ekip_ingestion_duration_s         histogram Per-document ingestion latency
  ekip_embedding_requests_total     counter   Embedding API calls by provider
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# ── HTTP layer ────────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    "ekip_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "ekip_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ── RAG pipeline ──────────────────────────────────────────────────────────────

RAG_REQUESTS_TOTAL = Counter(
    "ekip_rag_requests_total",
    "Total RAG ask requests",
    ["status"],   # success | error
)

RAG_RETRIEVAL_DURATION = Histogram(
    "ekip_rag_retrieval_duration_seconds",
    "Retrieval step duration (embed + vector search + rerank)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RAG_LLM_DURATION = Histogram(
    "ekip_rag_llm_duration_seconds",
    "LLM generation duration",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

RAG_CONTEXT_CHUNKS = Histogram(
    "ekip_rag_context_chunks",
    "Number of chunks in LLM context per request",
    buckets=[1, 2, 3, 5, 8, 10, 15, 20],
)

# ── Agent layer ───────────────────────────────────────────────────────────────

AGENT_REQUESTS_TOTAL = Counter(
    "ekip_agent_requests_total",
    "Total agent run requests",
    ["status"],
)

AGENT_ITERATIONS = Histogram(
    "ekip_agent_iterations",
    "Tool-call iterations per agent request",
    buckets=[1, 2, 3, 5, 7, 10],
)

AGENT_TOOL_CALLS_TOTAL = Counter(
    "ekip_agent_tool_calls_total",
    "Total agent tool calls",
    ["tool_name", "status"],
)

# ── Document ingestion ────────────────────────────────────────────────────────

INGESTION_DURATION = Histogram(
    "ekip_ingestion_duration_seconds",
    "Per-document ingestion duration (parse + chunk + embed + upsert)",
    ["file_type"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

INGESTION_TOTAL = Counter(
    "ekip_ingestion_total",
    "Total document ingestion attempts",
    ["file_type", "status"],   # status: success | failed
)

CHUNKS_PRODUCED = Histogram(
    "ekip_ingestion_chunks_produced",
    "Chunks produced per document",
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000],
)

# ── Corpus state (gauges — updated periodically by a background task) ─────────

DOCUMENTS_TOTAL = Gauge(
    "ekip_documents_total",
    "Total documents by status",
    ["status"],
)

CHUNKS_TOTAL = Gauge(
    "ekip_chunks_total",
    "Total indexed chunks in vector store",
)

# ── Embedding service ─────────────────────────────────────────────────────────

EMBEDDING_REQUESTS_TOTAL = Counter(
    "ekip_embedding_requests_total",
    "Total embedding API calls",
    ["provider"],
)

EMBEDDING_CACHE_HITS_TOTAL = Counter(
    "ekip_embedding_cache_hits_total",
    "Embedding cache hits (avoided API calls)",
    ["tier"],   # memory | redis
)
