"""
app/chunkers/semantic_chunker.py
─────────────────────────────────────────────────────────────────────────────
Semantic chunker — splits text at points of low embedding similarity.

Algorithm:
  1. Sentence-tokenise the input text (NLTK punkt, or regex fallback)
  2. Embed each sentence with the configured Embedder
  3. Compute cosine similarity between every adjacent pair of sentences
  4. Identify "breakpoints" where similarity < threshold
  5. Group consecutive sentences between breakpoints into chunks
  6. Merge small groups and split oversized ones (token-budget guardrail)

Best for:
  - Long documents where topic shifts are gradual (meeting notes, reports)
  - When you want RAG chunks to be semantically self-contained
  - Premium ingestion pipelines where accuracy > speed

Trade-offs vs Recursive:
  ✓ Better topic coherence per chunk
  ✓ Breakpoints follow meaning, not just punctuation
  ✗ Requires an embedding call during ingestion (slower, costs tokens)
  ✗ Threshold is a hyperparameter that needs tuning per corpus
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import List, Optional

from app.chunkers.token_counter import TokenCounter
from app.core.base.chunker import BaseChunker
from app.core.base.embedder import BaseEmbedder
from app.core.models.document import Chunk, ChunkingStrategy, Document

logger = logging.getLogger(__name__)

# Fallback sentence splitter when NLTK is unavailable
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class SemanticChunker(BaseChunker):
    """
    Splits documents at semantic breakpoints detected via embedding similarity.

    Args:
        chunk_size:         Hard token ceiling per chunk (default 512).
        chunk_overlap:      Overlap tokens between chunks (default 64).
        breakpoint_threshold: Cosine similarity below this → new chunk (default 0.85).
                              Lower = fewer, larger chunks.
                              Higher = more, smaller chunks.
        min_sentences:      Minimum sentences per chunk before forced merge (default 2).
        embedder:           BaseEmbedder instance. If None, uses EmbedderFactory.get()
                            which reads EMBEDDING_PROVIDER from config.
        tiktoken_model:     For token counting (default "cl100k_base").
    """

    strategy = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        chunk_size:           int   = 512,
        chunk_overlap:        int   = 64,
        breakpoint_threshold: float = 0.85,
        min_sentences:        int   = 2,
        embedder:             Optional[BaseEmbedder] = None,
        tiktoken_model:       str   = "cl100k_base",
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._threshold    = breakpoint_threshold
        self._min_sents    = min_sentences
        self._embedder     = embedder   # lazy-resolved in _get_embedder()
        self._tc           = TokenCounter(model=tiktoken_model)

    # ── BaseChunker interface ─────────────────────────────────────────────────

    def _chunk(self, text: str, document: Document) -> List[Chunk]:
        """
        Synchronous entry point (called by BaseChunker.chunk()).
        Runs the async embedding pipeline inside a new event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an async context (e.g. FastAPI) —
                # schedule as a coroutine and wait
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_chunk(text, document))
                    return future.result()
            else:
                return loop.run_until_complete(self._async_chunk(text, document))
        except RuntimeError:
            return asyncio.run(self._async_chunk(text, document))

    async def _async_chunk(self, text: str, document: Document) -> List[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Single sentence — no breakpoints possible
        if len(sentences) == 1:
            chunk = self._make_chunk(text=sentences[0], document=document)
            chunk.metadata.token_count = self._tc.count(sentences[0])
            return [chunk]

        # Embed all sentences in one batched call
        embedder = self._get_embedder()
        embeddings = await embedder.embed(sentences)

        # Find breakpoints
        groups = self._group_sentences(sentences, embeddings)

        # Build chunks from groups, enforcing token budget
        chunks: List[Chunk] = []
        for group in groups:
            group_text = " ".join(group).strip()
            if not group_text:
                continue

            # If group exceeds chunk_size, split it recursively
            if self._tc.count(group_text) > self.chunk_size:
                sub_chunks = self._split_oversized(group_text, document)
                chunks.extend(sub_chunks)
            else:
                chunk = self._make_chunk(text=group_text, document=document)
                chunk.metadata.token_count = self._tc.count(group_text)
                chunks.append(chunk)

        return chunks

    # ── Sentence splitting ────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences.
        Uses NLTK punkt tokeniser if available, else regex fallback.
        """
        try:
            import nltk
            try:
                tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
            sentences = tokenizer.tokenize(text.strip())
        except Exception:
            # Regex fallback: split on ". " / "! " / "? " followed by uppercase
            raw = _SENTENCE_RE.split(text.strip())
            sentences = [s.strip() for s in raw if s.strip()]

        return [s.strip() for s in sentences if s.strip()]

    # ── Breakpoint detection ──────────────────────────────────────────────────

    def _group_sentences(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
    ) -> List[List[str]]:
        """
        Group sentences into semantic chunks by finding cosine similarity
        breakpoints between adjacent sentence embeddings.
        """
        groups: List[List[str]] = []
        current_group: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            is_breakpoint = sim < self._threshold

            if is_breakpoint and len(current_group) >= self._min_sents:
                groups.append(current_group)
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])

        if current_group:
            groups.append(current_group)

        # Merge tiny groups (< min_sentences) into the previous group
        merged: List[List[str]] = []
        for group in groups:
            if merged and len(group) < self._min_sents:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged

    # ── Oversized group handling ──────────────────────────────────────────────

    def _split_oversized(self, text: str, document: Document) -> List[Chunk]:
        """
        When a semantic group exceeds chunk_size, fall back to token-window
        splitting within that group to keep the token budget.
        """
        from app.chunkers.recursive_chunker import RecursiveChunker
        splitter = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        # Produce raw text pieces, then wrap as Chunks
        pieces = splitter._split_recursive(text, splitter._separators)
        merged = splitter._merge_with_overlap(pieces)
        chunks = []
        for t in merged:
            if not t.strip():
                continue
            chunk = self._make_chunk(text=t.strip(), document=document)
            chunk.metadata.token_count = self._tc.count(t)
            chunks.append(chunk)
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            from app.core.base.embedder import EmbedderFactory
            self._embedder = EmbedderFactory.get()
        return self._embedder

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
