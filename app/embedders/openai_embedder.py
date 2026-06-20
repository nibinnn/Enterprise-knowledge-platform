"""
app/embedders/openai_embedder.py
─────────────────────────────────────────────────────────────────────────────
OpenAI embedding provider.

Supported models:
  text-embedding-3-small   1536 dims   ~$0.02 / 1M tokens  (recommended)
  text-embedding-3-large   3072 dims   ~$0.13 / 1M tokens  (best quality)
  text-embedding-ada-002   1536 dims   legacy

Features:
  - Async batching via BaseEmbedder (up to `batch_size` texts per API call)
  - Exponential-backoff retry on rate-limit / transient errors (tenacity)
  - Redis + in-memory two-tier cache (EmbeddingCache)
  - Automatic text truncation at 8 191 tokens (model limit)
  - L2 normalisation so dot-product == cosine similarity downstream
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.base.embedder import BaseEmbedder
from app.embedders.cache import EmbeddingCache

logger = logging.getLogger(__name__)

# Model → output dimension
_MODEL_DIMS: Dict[str, int] = {
    "text-embedding-3-small":  1536,
    "text-embedding-3-large":  3072,
    "text-embedding-ada-002":  1536,
}

# OpenAI hard token limit per text
_MAX_TOKENS = 8_191


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding provider.

    Args:
        model:         OpenAI embedding model name.
        api_key:       OpenAI API key. Falls back to OPENAI_API_KEY env var.
        batch_size:    Max texts per API call (OpenAI limit: 2048; default 100).
        cache_enabled: Enable in-memory session cache (BaseEmbedder layer).
        redis_url:     Redis connection string for persistent cache.
                       None = memory-only cache.
        dimensions:    Output dimension override for text-embedding-3-* models
                       (supports Matryoshka dimensionality reduction).
    """

    provider_name = "openai"

    def __init__(
        self,
        model:         str = "text-embedding-3-small",
        api_key:       Optional[str] = None,
        batch_size:    int = 100,
        cache_enabled: bool = True,
        redis_url:     Optional[str] = None,
        dimensions:    Optional[int] = None,
    ):
        super().__init__(batch_size=batch_size, cache_enabled=cache_enabled)

        self.model_name = model
        self.dimension  = dimensions or _MODEL_DIMS.get(model, 1536)
        self._dimensions = dimensions   # for Matryoshka support

        # API key: explicit → env var
        self._api_key = api_key or self._get_api_key()

        # Persistent cache
        self._cache_layer = EmbeddingCache(
            provider="openai", model=model, redis_url=redis_url
        )

        # Lazy client — instantiated on first use
        self._client = None

    # ── BaseEmbedder interface ────────────────────────────────────────────────

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed one batch, checking the Redis cache before hitting the API."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts:   List[str] = []

        # Check persistent cache first
        for i, text in enumerate(texts):
            cached = self._cache_layer.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = await self._call_api(uncached_texts)
            for idx, vec in zip(uncached_indices, embeddings):
                normalised      = self._normalise(vec)
                results[idx]    = normalised
                self._cache_layer.set(texts[idx], normalised)

        return results  # type: ignore[return-value]

    async def embed_query(self, query: str) -> List[float]:
        """
        OpenAI uses the same model for queries and documents —
        no special prefix needed (unlike Cohere / E5 models).
        """
        embeddings = await self.embed([query])
        return embeddings[0]

    # ── API call with retry ───────────────────────────────────────────────────

    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API with automatic retry on transient errors."""
        import asyncio

        try:
            from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        @retry(
            retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        async def _request(client, payload_texts):
            kwargs = dict(input=payload_texts, model=self.model_name)
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = await client.embeddings.create(**kwargs)
            return [item.embedding for item in response.data]

        client = self._get_client()
        try:
            logger.debug("[OpenAI] Embedding %d texts", len(texts))
            return await _request(client, [self._truncate(t) for t in texts])
        except Exception as exc:
            logger.error("[OpenAI] Embedding failed: %s", exc)
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @staticmethod
    def _truncate(text: str, max_chars: int = _MAX_TOKENS * 4) -> str:
        """
        Rough character-based truncation (1 token ≈ 4 chars).
        The real truncation happens server-side; this avoids sending
        extremely long strings that waste bandwidth.
        """
        return text[:max_chars] if len(text) > max_chars else text

    @staticmethod
    def _get_api_key() -> str:
        from app.config import get_settings
        key = get_settings().openai_api_key
        if not key:
            raise ValueError(
                "OpenAI API key not set. Add OPENAI_API_KEY to your .env file."
            )
        return key
