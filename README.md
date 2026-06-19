# Enterprise Knowledge Intelligence Platform

An end-to-end GenAI application that enables organizations to ingest, process, search, and analyze enterprise knowledge using Retrieval-Augmented Generation (RAG), AI Agents, Vector Search, and Hybrid Retrieval.

## Overview

Organizations store critical information across PDFs, Word documents, policies, SOPs, manuals, wiki pages, and reports. Finding relevant information is often slow and inefficient.

This project provides an AI-powered knowledge platform that:

- Ingests and processes enterprise documents
- Cleans and structures raw data
- Generates embeddings for semantic understanding
- Stores vectors in a vector database
- Performs hybrid retrieval (Vector + BM25)
- Uses AI agents for complex reasoning
- Produces grounded responses with citations
- Evaluates retrieval and generation quality

---

## Key Features

### Document Ingestion

**Supported Formats**

- PDF
- DOCX
- TXT
- HTML
- Markdown

**Features**

- Metadata extraction
- OCR-ready architecture
- Batch document processing

---

### Data Cleaning & Preprocessing

The preprocessing pipeline includes:

- Header removal
- Footer removal
- Duplicate content elimination
- Page number removal
- Special character cleanup
- Metadata enrichment

---

### Advanced Chunking Strategies

Implemented chunking methods:

#### Fixed Chunking

- Fixed token size
- Configurable overlap

#### Recursive Chunking

- Paragraph-level splitting
- Sentence-level splitting
- Token-level fallback

#### Semantic Chunking

- Embedding-based chunk boundaries
- Context-aware segmentation

---

### Embedding Generation

Supports pluggable embedding providers:

- OpenAI Embeddings
- BAAI BGE Models
- E5 Models
- Instructor Embeddings

---

### Vector Database

Compatible with:

- ChromaDB
- Qdrant
- FAISS

**Capabilities**

- Semantic Search
- Similarity Search
- Metadata Filtering

---

### Hybrid Retrieval

Combines:

- Dense Retrieval (Vector Search)
- Sparse Retrieval (BM25)

**Benefits**

- Better recall
- Improved relevance
- Reduced hallucinations

---

### Re-ranking Layer

Uses Cross Encoder models to:

- Re-rank retrieved chunks
- Improve context quality
- Increase answer accuracy

---

### Retrieval-Augmented Generation (RAG)

#### Pipeline

```text
User Query
    ↓
Query Processing
    ↓
Hybrid Retrieval
    ↓
Re-ranking
    ↓
Context Assembly
    ↓
LLM Response
    ↓
Citation Generation
```

---

### AI Agents

Built using **LangGraph**.

#### Available Agents

##### Search Agent

Retrieves relevant information from the knowledge base.

##### Summarization Agent

Generates concise document summaries.

##### Comparison Agent

Compares information across multiple documents.

##### Research Agent

Produces detailed reports from multiple sources.

##### Routing Agent

Routes user requests to the appropriate workflow.

---

### Source Citations

Every answer includes:

- Source document
- Page number
- Chunk reference

---

## Tech Stack

- Python
- FastAPI
- LangChain
- LangGraph
- ChromaDB
- Qdrant
- FAISS
- RAGAS
- Streamlit

---

## License

MIT License