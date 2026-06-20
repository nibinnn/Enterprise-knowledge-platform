"""
app/embedders/cohere_embedder.py
─────────────────────────────────────────────────────────────────────────────
Cohere embedding provider.

Key difference from OpenAI: Cohere requires a different `input_type` for
documents vs queries — this is mandatory, not optional.

  input_type="search_document"  → used during ingestion (chunk embedding)
  input_type="search_query"     → used at query time (embed_query)

Supported models:
  embed-english-v3.0         1024 dims  (best English quality)
  embed-multilingual-v3.0    1024 dims  (multilingual)
  embed-english-light-v3.0    384 dims  (faster, smaller)
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

_MODEL_DIMS: Dict[str, int] = {
    "embed-english-v3.0":          1024,
    "embed-multilingual-v3.0":     1024,
    "embed-english-light-v3.0":     384,
    "embed-multilingual-light-v3.0": 384,
}

# Cohere max texts per request
_MAX_BATCH = 96


class CohereEmbedder(BaseEmbedder):
    """
    Cohere embedding provider.

    Args:
        model:         Cohere embedding model name.
        api_key:       Cohere API key. Falls back to COHERE_API_KEY env var.
        batch_size:    Max texts per API call (Cohere limit: 96).
        cache_enabled: Enable in-memory session cache.
        redis_url:     Redis URL for persistent cache.
    """

    provider_name = "cohere"

    def __init__(
        self,
        model:         str = "embed-english-v3.0",
        api_key:       Optional[str] = None,
        batch_size:    int = 96,
        cache_enabled: bool = True,
        redis_url:     Optional[str] = None,
    ):
        # Cohere batch limit is 96 — clamp silently
        batch_size = min(batch_size, _MAX_BATCH)
        super().__init__(batch_size=batch_size, cache_enabled=cache_enabled)

        self.model_name = model
        self.dimension  = _MODEL_DIMS.get(model, 1024)
        self._api_key   = api_key or self._get_api_key()

        self._cache_layer = EmbeddingCache(
            provider="cohere", model=model, redis_url=redis_url
        )
        self._client = None

    # ── BaseEmbedder interface ────────────────────────────────────────────────

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed one batch as *search_document* (ingestion path)."""
        return await self._embed_with_type(texts, input_type="search_document")

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query using Cohere's *search_query* input type.
        This is different from document embedding — Cohere uses separate
        model heads for queries and documents.
        """
        results = await self._embed_with_type([query], input_type="search_query")
        return results[0]

    # ── Shared embed implementation ───────────────────────────────────────────

    async def _embed_with_type(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        """Check cache, call API, normalise, store — for a given input_type."""
        # Use input_type as part of cache key suffix to prevent collisions
        # between query and document embeddings of the same text
        cache_texts = [f"{input_type}:{t}" for t in texts]

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts:   List[str] = []

        for i, ct in enumerate(cache_texts):
            cached = self._cache_layer.get(ct)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(texts[i])

        if uncached_texts:
            embeddings = await self._call_api(uncached_texts, input_type)
            for idx, vec in zip(uncached_indices, embeddings):
                normalised   = self._normalise(vec)
                results[idx] = normalised
                self._cache_layer.set(cache_texts[idx], normalised)

        return results  # type: ignore[return-value]

    # ── API call with retry ───────────────────────────────────────────────────

    async def _call_api(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        import asyncio

        try:
            import cohere
        except ImportError:
            raise RuntimeError("cohere package not installed. Run: pip install cohere")

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        def _sync_request(client, payload_texts, itype):
            response = client.embed(
                texts=payload_texts,
                model=self.model_name,
                input_type=itype,
            )
            return [list(e) for e in response.embeddings]

        client = self._get_client()
        logger.debug("[Cohere] Embedding %d texts (input_type=%s)", len(texts), input_type)

        loop = asyncio.get_event_loop()
        try:
            # Cohere SDK is synchronous — run in thread pool
            return await loop.run_in_executor(
                None, _sync_request, client, texts, input_type
            )
        except Exception as exc:
            logger.error("[Cohere] Embedding failed: %s", exc)
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _get_api_key() -> str:
        from app.config import get_settings
        key = get_settings().cohere_api_key
        if not key:
            raise ValueError(
                "Cohere API key not set. Add COHERE_API_KEY to your .env file."
            )
        return key
