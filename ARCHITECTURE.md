# ARCHITECTURE.md — Enterprise Knowledge Intelligence Platform

> **Keep this file open in every session. Paste it into new chats as context.**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EKIP — System Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────────┐  │
│  │ Document │    │ Chunking │    │ Embedding │    │  Vector Store   │  │
│  │  Parser  │───▶│ Pipeline │───▶│  Service  │───▶│    (Qdrant)     │  │
│  └──────────┘    └──────────┘    └───────────┘    └────────┬────────┘  │
│       │                                                     │           │
│  (PDF, DOCX,                                         ┌──────┴──────┐   │
│   TXT, MD,                                           │   Hybrid    │   │
│   HTML, …)                                           │   Search    │   │
│                                                      │ Dense+KW+RRF│   │
│                                                      └──────┬──────┘   │
│                                                             │           │
│  ┌──────────────────────────────────────────────────────────▼────────┐ │
│  │                        RAG Pipeline                               │ │
│  │  Query → Retriever → Re-ranker → Context Builder → LLM → Answer  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      AI Agent Layer                              │  │
│  │  SearchTool | DocumentTool | SummarizationTool | AnalyticsTool  │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │  Agentic Research Mode: Query Decomp → Multi-step RAG   │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────────┐ │
│  │   Citation   │  │  Evaluation  │  │         FastAPI + Streamlit   │ │
│  │   Engine     │  │  Framework   │  │              UI               │ │
│  └──────────────┘  └──────────────┘  └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
                            ┌───────────────┐
                            │  File Upload  │
                            └──────┬────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   Celery Job Queue  │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │               Ingestion Worker           │
              │                                          │
              │  1. Parser (PDF/DOCX/TXT/MD/HTML)        │
              │       → Document (raw_text + sections)   │
              │                                          │
              │  2. Chunker (fixed|recursive|semantic)   │
              │       → List[Chunk]                      │
              │                                          │
              │  3. Embedder (OpenAI|Cohere|HF)          │
              │       → List[EmbeddedChunk]              │
              │                                          │
              │  4. VectorStore.upsert()                 │
              │       → Qdrant (dense) + BM25 index      │
              │                                          │
              │  5. DB: documents.status = "indexed"     │
              └─────────────────────────────────────────┘

                          Query Pipeline

              ┌────────────────────────────────────────┐
              │                                        │
              │  User Query                            │
              │    │                                   │
              │    ▼                                   │
              │  [optional] Query Rewriter             │
              │    │                                   │
              │    ▼                                   │
              │  Retriever                             │
              │    ├── Dense Search (Qdrant ANN)       │
              │    ├── Keyword Search (BM25)           │
              │    └── Hybrid Fusion (RRF)             │
              │    │                                   │
              │    ▼                                   │
              │  Re-ranker (cross-encoder)             │
              │    │                                   │
              │    ▼                                   │
              │  Context Builder                       │
              │    (deduplicate + token-budget trim)   │
              │    │                                   │
              │    ▼                                   │
              │  LLM (Claude / GPT-4)                  │
              │    │                                   │
              │    ▼                                   │
              │  Answer + Citations                    │
              └────────────────────────────────────────┘
```

---

## Module Map

| Module | Location | Built | Purpose |
|--------|----------|-------|---------|
| Config | `app/config.py` | Day 1 | All env-var settings |
| Domain models | `app/core/models/` | Day 1 | Document, Chunk, Answer, … |
| Abstract base classes | `app/core/base/` | Day 1 | Interfaces for all pluggable components |
| DB schema | `app/db/` | Day 1 | SQLAlchemy models + SQL init |
| PDF Parser | `app/parsers/pdf_parser.py` | Day 2 | PyMuPDF + pdfplumber + OCR |
| DOCX / multi-format | `app/parsers/` | Day 3 | python-docx, unstructured |
| Fixed / Recursive Chunker | `app/chunkers/` | Day 4 | Token-based chunking |
| Semantic Chunker | `app/chunkers/semantic_chunker.py` | Day 5 | Embedding-breakpoint chunking |
| Embedding Service | `app/embedders/` | Day 6 | OpenAI, Cohere, SentenceTransformers |
| Qdrant VectorStore | `app/vector_store/qdrant_store.py` | Day 7 | Upsert + search |
| Dense Search | `app/retrieval/dense.py` | Day 8 | ANN search |
| Keyword Search | `app/retrieval/keyword.py` | Day 9 | BM25 |
| Hybrid Search | `app/retrieval/hybrid.py` | Day 10 | RRF fusion |
| Retriever | `app/retrieval/retriever.py` | Day 11 | Query rewrite + search |
| Re-ranker | `app/retrieval/reranker.py` | Day 12 | Cross-encoder |
| Context Builder | `app/rag/context_builder.py` | Day 13 | Dedup + token budget |
| LLM + multi-modal | `app/rag/llm.py` | Day 14 | Answer generation |
| RAG Pipeline | `app/rag/pipeline.py` | Day 15 | End-to-end wiring |
| Agent + SearchTool + DocumentTool | `app/agents/` | Day 16 | Tool-calling agent |
| SummarizationTool + AnalyticsTool | `app/agents/tools/` | Day 17 | Agent tools |
| Research mode + feedback | `app/agents/research.py` | Day 18 | Multi-step reasoning |
| Citation Engine | `app/citations/` | Day 19 | Source tracking |
| Evaluation | `app/evaluation/` | Day 20 | RAGAS metrics |
| FastAPI routes | `app/api/routes/` | Day 21 | REST API |
| Frontend | `frontend/` | Day 22 | Streamlit UI |
| Async ingestion | `app/workers/` | Day 23 | Celery tasks |
| Docker / deploy | `docker/` | Day 24 | Production containers |

---

## Abstract Interfaces (stable — all days depend on these)

```python
# Parser
class BaseParser:
    supported_extensions: frozenset[str]
    def parse(file_path) -> Document: ...

# Chunker
class BaseChunker:
    def chunk(document: Document) -> List[Chunk]: ...

# Embedder
class BaseEmbedder:
    async def embed(texts: List[str]) -> List[List[float]]: ...
    async def embed_query(query: str) -> List[float]: ...

# VectorStore
class BaseVectorStore:
    async def upsert(chunks: List[EmbeddedChunk]) -> int: ...
    async def search(query_embedding, top_k, filters) -> List[SearchResult]: ...
    async def hybrid_search(...) -> List[SearchResult]: ...

# Retriever
class BaseRetriever:
    async def retrieve(query, top_k, filters) -> List[RetrievedChunk]: ...

# Tool
class BaseTool:
    name: str
    description: str
    parameters_schema: dict
    def run(**kwargs) -> ToolResult: ...
```

---

## Tech Stack Decisions

| Concern | Choice | Reason |
|---------|--------|--------|
| Web framework | FastAPI | Async-native, auto-docs, Pydantic integration |
| Database | PostgreSQL 16 | JSONB, ARRAY types, mature ACID semantics |
| ORM | SQLAlchemy 2.0 async | Type-safe mapped columns, async-first |
| Vector DB | Qdrant | Native hybrid search, on-premise, fast |
| Keyword search | BM25 (rank_bm25) | Simple, no extra infra needed |
| Embedding | OpenAI (pluggable) | Best quality; Cohere/HF as drop-in alternatives |
| LLM | Claude (pluggable) | Strong context window, citation quality |
| Task queue | Celery + Redis | Proven, simple broker, async ingestion |
| Agent framework | LangGraph | Explicit graph, easy to debug |
| Evaluation | RAGAS | Industry-standard RAG metrics |

---

## Key Design Principles

1. **Interfaces before implementations.** All concrete classes inherit from abstract bases defined on Day 1. Swap any provider without touching upstream code.
2. **Metadata flows everywhere.** Every Chunk carries `doc_id`, `filename`, `page_number`, `section`, `department` — so citations are always traceable.
3. **One session per day.** Each day's work is a separate Claude conversation. Start each with `ARCHITECTURE.md` + the previous `PROGRESS.md`.
4. **Async-first.** FastAPI routes, DB sessions, and embedding calls are all `async`. Celery workers use sync sessions for simplicity.
5. **Fail loudly in dev, gracefully in prod.** Parsing errors raise `ParseError`; the ingestion job catches them, updates `jobs.status = "failed"`, and continues with the next document.
