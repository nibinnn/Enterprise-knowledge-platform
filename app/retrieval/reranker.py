"""
app/retrieval/reranker.py
─────────────────────────────────────────────────────────────────────────────
Re-ranking layer: takes the top-N retrieved chunks and re-orders them
using a more expensive but more accurate cross-encoder model.

Two backends:
  CrossEncoderReranker  — local sentence-transformers cross-encoder (free)
  CohereReranker        — Cohere Rerank API (higher accuracy, costs tokens)
  NoOpReranker          — pass-through (for testing / when reranking disabled)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from app.core.models.document import RetrievedChunk

logger = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")


class NoOpReranker:
    """Pass-through — returns chunks in their original order."""
    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        reranked = chunks[:top_n]
        for i, c in enumerate(reranked, 1):
            c.rank = i
        return reranked


class CrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence-transformers.
    Default model: ms-marco-MiniLM-L-6-v2 (fast, good quality)
    """
    _model_cache: dict = {}

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model
        self._model = None

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        loop = asyncio.get_event_loop()
        scored = await loop.run_in_executor(_POOL, self._score_sync, query, chunks)
        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        for rank, (chunk, score) in enumerate(scored[:top_n], 1):
            chunk.rerank_score = float(score)
            chunk.score = float(score)
            chunk.rank = rank
            result.append(chunk)
        return result

    def _score_sync(self, query: str, chunks: List[RetrievedChunk]):
        model = self._get_model()
        pairs  = [[query, c.text] for c in chunks]
        scores = model.predict(pairs)
        return list(zip(chunks, scores))

    def _get_model(self):
        if self._model_name not in self._model_cache:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder '%s'…", self._model_name)
            self._model_cache[self._model_name] = CrossEncoder(self._model_name)
        return self._model_cache[self._model_name]


class CohereReranker:
    """Cohere Rerank API — higher accuracy, requires API key."""

    def __init__(self, model: str = "rerank-english-v3.0", api_key: Optional[str] = None):
        self._model   = model
        self._api_key = api_key or self._get_key()
        self._client  = None

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_POOL, self._rerank_sync, query, chunks, top_n)

    def _rerank_sync(self, query: str, chunks: List[RetrievedChunk], top_n: int):
        import cohere
        client   = cohere.Client(api_key=self._api_key)
        docs     = [c.text for c in chunks]
        response = client.rerank(query=query, documents=docs, model=self._model, top_n=top_n)
        result   = []
        for rank, hit in enumerate(response.results, 1):
            chunk = chunks[hit.index]
            chunk.rerank_score = hit.relevance_score
            chunk.score        = hit.relevance_score
            chunk.rank         = rank
            result.append(chunk)
        return result

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _get_key() -> str:
        from app.config import get_settings
        return get_settings().cohere_api_key


def get_reranker(provider: str = "cross_encoder", **kwargs):
    """Factory — returns the configured reranker."""
    if provider == "cohere":
        return CohereReranker(**kwargs)
    if provider == "none":
        return NoOpReranker()
    return CrossEncoderReranker(**kwargs)
