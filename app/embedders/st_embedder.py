"""
app/embedders/st_embedder.py
─────────────────────────────────────────────────────────────────────────────
SentenceTransformers embedding provider (100% local, no API calls).

Use this when:
  - Data cannot leave your infrastructure
  - You want zero API cost during development / testing
  - You have GPU capacity and want high throughput

Recommended models (trade-off: speed vs quality):
  all-MiniLM-L6-v2          384 dims  fast,  good for most enterprise RAG
  all-mpnet-base-v2          768 dims  slower, better semantic quality
  BAAI/bge-small-en-v1.5    384 dims  great for RAG (BGE family, instruction-tuned)
  BAAI/bge-large-en-v1.5   1024 dims  best quality, needs more RAM

Design choices:
  - Model is loaded ONCE and cached across all instances via class-level dict
  - Inference is synchronous (torch) — run in thread pool to stay async
  - encode() is called with `normalize_embeddings=True` so L2 norm is done by
    the library instead of our `_normalise()` — same result, faster
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from app.core.base.embedder import BaseEmbedder
from app.embedders.cache import EmbeddingCache

logger = logging.getLogger(__name__)

# Model → output dimension (common models)
_MODEL_DIMS: Dict[str, int] = {
    "all-MiniLM-L6-v2":          384,
    "all-MiniLM-L12-v2":         384,
    "all-mpnet-base-v2":          768,
    "paraphrase-MiniLM-L6-v2":   384,
    "BAAI/bge-small-en-v1.5":    384,
    "BAAI/bge-base-en-v1.5":     768,
    "BAAI/bge-large-en-v1.5":   1024,
}

# Shared thread pool for running synchronous torch inference
_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="st_embed")

# Class-level model cache — avoids reloading the same model multiple times
_MODEL_CACHE: Dict[str, object] = {}


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Local embedding provider using HuggingFace SentenceTransformers.

    Args:
        model:         HuggingFace model name or local path.
        device:        "cpu", "cuda", or "auto" (auto-detects GPU).
        batch_size:    Texts per encode() call.
        cache_enabled: In-memory session cache (BaseEmbedder layer).
        redis_url:     Redis URL for persistent cache.
        show_progress: Show tqdm progress during encoding (default False).
    """

    provider_name = "sentence_transformers"

    def __init__(
        self,
        model:          str = "all-MiniLM-L6-v2",
        device:         str = "auto",
        batch_size:     int = 64,
        cache_enabled:  bool = True,
        redis_url:      Optional[str] = None,
        show_progress:  bool = False,
    ):
        super().__init__(batch_size=batch_size, cache_enabled=cache_enabled)

        self.model_name    = model
        self._device_pref  = device
        self._show_progress = show_progress

        self.dimension = _MODEL_DIMS.get(model, 0)  # 0 = unknown until first encode

        self._cache_layer = EmbeddingCache(
            provider="sentence_transformers", model=model, redis_url=redis_url
        )
        self._st_model = None   # lazy-loaded on first call

    # ── BaseEmbedder interface ────────────────────────────────────────────────

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Check cache, then run synchronous ST inference in a thread pool."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts:   List[str] = []

        for i, text in enumerate(texts):
            cached = self._cache_layer.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = await self._encode_async(uncached_texts)
            for idx, vec in zip(uncached_indices, embeddings):
                results[idx] = vec
                self._cache_layer.set(texts[idx], vec)

        return results  # type: ignore[return-value]

    async def embed_query(self, query: str) -> List[float]:
        """
        BGE models expect a query prefix: "Represent this sentence: ..."
        Other models work fine without it.
        """
        prefixed = self._maybe_add_query_prefix(query)
        results  = await self._encode_async([prefixed])
        return results[0]

    # ── Async inference ───────────────────────────────────────────────────────

    async def _encode_async(self, texts: List[str]) -> List[List[float]]:
        """Run synchronous SentenceTransformer.encode() in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _THREAD_POOL,
            self._encode_sync,
            texts,
        )

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous encode — runs inside the thread pool.
        normalize_embeddings=True means ST does L2 normalisation for us.
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in embeddings]

    # ── Model loading ─────────────────────────────────────────────────────────

    def _get_model(self):
        """Return the cached SentenceTransformer model, loading if necessary."""
        if self.model_name not in _MODEL_CACHE:
            _MODEL_CACHE[self.model_name] = self._load_model()
        self._st_model = _MODEL_CACHE[self.model_name]
        return self._st_model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        device = self._resolve_device()
        logger.info(
            "[SentenceTransformers] Loading model '%s' on device '%s'…",
            self.model_name, device,
        )
        model = SentenceTransformer(self.model_name, device=device)

        # Update dimension from the actual model
        self.dimension = model.get_sentence_embedding_dimension()
        logger.info(
            "[SentenceTransformers] Model loaded. Dimension: %d", self.dimension
        )
        return model

    def _resolve_device(self) -> str:
        if self._device_pref != "auto":
            return self._device_pref
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _maybe_add_query_prefix(self, query: str) -> str:
        """
        BGE models perform better with an instruction prefix for queries.
        Other models are unaffected.
        """
        bge_prefixes = ("BAAI/bge-", "bge-")
        if any(self.model_name.startswith(p) for p in bge_prefixes):
            return f"Represent this sentence for retrieval: {query}"
        return query

    @property
    def is_model_loaded(self) -> bool:
        return self.model_name in _MODEL_CACHE
