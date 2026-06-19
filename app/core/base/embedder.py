"""
app/core/base/embedder.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for the embedding service.

Pluggable architecture — swap providers by changing EMBEDDING_PROVIDER env var:
  - OpenAIEmbedder          (text-embedding-3-small / large)
  - CohereEmbedder          (embed-english-v3.0)
  - SentenceTransformersEmbedder  (local, GPU-optional)

All three will be implemented on Day 6.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Contract every embedding provider must fulfil.

    Key design decisions:
    - `embed()` is async and batches internally → safe to call with thousands of texts
    - `embed_query()` is a thin alias that adds query-specific prefixes where needed
    - All embeddings are L2-normalised (unit vectors) so dot-product == cosine similarity
    - An in-memory cache prevents re-embedding unchanged chunks within a session
    """

    # Subclasses set these class-level attributes
    provider_name: str = "base"
    model_name: str = ""
    dimension: int = 0

    def __init__(self, batch_size: int = 100, cache_enabled: bool = True):
        self.batch_size = batch_size
        self._cache: Dict[str, List[float]] = {} if cache_enabled else {}
        self._cache_enabled = cache_enabled

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts (document chunks).
        Returns a list of float vectors in the same order as the input.
        Handles batching and caching internally.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # 1. Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if self._cache_enabled and key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 2. Embed uncached texts in batches
        if uncached_texts:
            logger.debug(
                "[%s] Embedding %d texts (%d from cache, %d new)",
                self.provider_name, len(texts),
                len(texts) - len(uncached_texts), len(uncached_texts),
            )
            batches = [
                uncached_texts[i: i + self.batch_size]
                for i in range(0, len(uncached_texts), self.batch_size)
            ]
            all_embeddings: List[List[float]] = []
            for batch in batches:
                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)

            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                if self._cache_enabled:
                    self._cache[self._cache_key(texts[idx])] = embedding

        return results  # type: ignore[return-value]

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query.
        Some providers (Cohere, E5 models) use different prefixes
        for queries vs documents — subclasses can override this.
        """
        embeddings = await self.embed([query])
        return embeddings[0]

    async def embed_chunks(self, chunks) -> List[List[float]]:
        """Convenience: embed a list of Chunk objects."""
        texts = [chunk.text for chunk in chunks]
        return await self.embed(texts)

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ── To implement ──────────────────────────────────────────────────────────

    @abstractmethod
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a single batch of texts.
        The public `embed()` handles batching, caching, and logging —
        implement only the provider API call here.
        """

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _normalise(vector: List[float]) -> List[float]:
        """L2-normalise a vector so dot-product == cosine similarity."""
        import math
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]


# ─────────────────────────── Factory ─────────────────────────────────────────

class EmbedderFactory:
    """
    Instantiates the correct embedder based on config.
    Concrete embedders are lazy-imported (built on Day 6).
    """

    @staticmethod
    def get(provider: Optional[str] = None, **kwargs) -> BaseEmbedder:
        """
        Args:
            provider: "openai" | "cohere" | "sentence_transformers"
                      Defaults to EMBEDDING_PROVIDER env var.
        Returns:
            A configured BaseEmbedder subclass instance.
        """
        from app.config import get_settings
        settings = get_settings()
        provider = provider or settings.embedding_provider.value

        if provider == "openai":
            from app.embedders.openai_embedder import OpenAIEmbedder
            return OpenAIEmbedder(
                model=kwargs.get("model", settings.embedding_model),
                batch_size=kwargs.get("batch_size", settings.embedding_batch_size),
            )
        if provider == "cohere":
            from app.embedders.cohere_embedder import CohereEmbedder
            return CohereEmbedder(
                batch_size=kwargs.get("batch_size", settings.embedding_batch_size),
            )
        if provider == "sentence_transformers":
            from app.embedders.st_embedder import SentenceTransformersEmbedder
            return SentenceTransformersEmbedder(
                model=kwargs.get("model", "all-MiniLM-L6-v2"),
                batch_size=kwargs.get("batch_size", settings.embedding_batch_size),
            )

        raise ValueError(f"Unknown embedding provider: '{provider}'")
