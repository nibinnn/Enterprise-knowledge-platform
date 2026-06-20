"""
app/embedders/cache.py
─────────────────────────────────────────────────────────────────────────────
Two-tier embedding cache used by all three provider implementations.

Tier 1 — In-memory dict  : zero-latency; lives for the process lifetime
Tier 2 — Redis           : survives restarts; shared across workers/replicas

Read order  :  memory → Redis → API call
Write order :  API result → Redis → memory

If Redis is unavailable (not configured or connection fails), the cache
degrades gracefully to memory-only without raising exceptions.

Key format : ekip:emb:{provider}:{model}:{md5(text)}
Value      : JSON-encoded List[float]
TTL        : configurable (default 24 h)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Persistent embedding cache backed by Redis with an in-memory hot layer.

    Args:
        provider:   Embedder provider name  (e.g. "openai").
        model:      Model identifier        (e.g. "text-embedding-3-small").
        redis_url:  Redis connection string. Pass None to disable Redis tier.
        ttl:        Redis key expiry in seconds (default 86400 = 24 h).
    """

    _KEY_PREFIX = "ekip:emb"

    def __init__(
        self,
        provider:  str,
        model:     str,
        redis_url: Optional[str] = None,
        ttl:       int = 86_400,
    ):
        self._provider  = provider
        self._model     = model
        self._ttl       = ttl
        self._memory:   Dict[str, List[float]] = {}
        self._redis     = self._connect(redis_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, text: str) -> Optional[List[float]]:
        """Return the cached embedding for *text*, or None if not cached."""
        key = self._key(text)

        # Tier 1: memory
        if key in self._memory:
            return self._memory[key]

        # Tier 2: Redis
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw:
                    vec = json.loads(raw)
                    self._memory[key] = vec   # promote to memory tier
                    return vec
            except Exception as exc:
                logger.debug("Redis cache get failed: %s", exc)

        return None

    def set(self, text: str, vector: List[float]) -> None:
        """Store an embedding in both tiers."""
        key = self._key(text)
        self._memory[key] = vector

        if self._redis is not None:
            try:
                self._redis.setex(key, self._ttl, json.dumps(vector))
            except Exception as exc:
                logger.debug("Redis cache set failed: %s", exc)

    def get_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Return cached embeddings for a list of texts (None = cache miss)."""
        return [self.get(t) for t in texts]

    def set_batch(self, texts: List[str], vectors: List[List[float]]) -> None:
        """Store a batch of embeddings."""
        for text, vec in zip(texts, vectors):
            self.set(text, vec)

    def clear_memory(self) -> None:
        """Clear only the in-memory tier (Redis is unaffected)."""
        self._memory.clear()

    @property
    def memory_size(self) -> int:
        return len(self._memory)

    @property
    def redis_available(self) -> bool:
        return self._redis is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _key(self, text: str) -> str:
        digest = hashlib.md5(text.encode()).hexdigest()
        return f"{self._KEY_PREFIX}:{self._provider}:{self._model}:{digest}"

    @staticmethod
    def _connect(redis_url: Optional[str]):
        """Return a Redis client, or None if unavailable/unconfigured."""
        if not redis_url:
            return None
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, decode_responses=True, socket_timeout=2)
            client.ping()
            logger.info("EmbeddingCache: Redis connected at %s", redis_url)
            return client
        except Exception as exc:
            logger.warning(
                "EmbeddingCache: Redis unavailable (%s) — using memory-only cache.", exc
            )
            return None
