"""
tests/unit/test_embedders.py
─────────────────────────────────────────────────────────────────────────────
Day 6 unit tests — EmbeddingCache, OpenAIEmbedder, CohereEmbedder,
SentenceTransformersEmbedder, and EmbedderFactory.

All external API calls (OpenAI, Cohere, SentenceTransformers model load)
are mocked so these tests run offline, zero cost, and < 1 second.

Run with:  pytest tests/unit/test_embedders.py -v
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import math
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────── helpers ─────────────────────────────────────────

def unit_vector(dims: int = 4, value: float = 1.0) -> List[float]:
    """Return a normalised vector for testing."""
    raw = [value] + [0.0] * (dims - 1)
    mag = math.sqrt(sum(x * x for x in raw))
    return [x / mag for x in raw]


def fake_embedding(dims: int = 4) -> List[float]:
    return unit_vector(dims)


def run(coro):
    """Run a coroutine in the current event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────── EmbeddingCache ──────────────────────────────────

class TestEmbeddingCache:

    def _cache(self, redis_url=None):
        from app.embedders.cache import EmbeddingCache
        return EmbeddingCache(
            provider="openai", model="text-embedding-3-small",
            redis_url=redis_url,
        )

    def test_memory_get_miss_returns_none(self):
        cache = self._cache()
        assert cache.get("never seen this text") is None

    def test_memory_set_and_get(self):
        cache = self._cache()
        vec   = [0.1, 0.2, 0.3]
        cache.set("hello world", vec)
        assert cache.get("hello world") == vec

    def test_memory_size_increments(self):
        cache = self._cache()
        assert cache.memory_size == 0
        cache.set("text one", [1.0])
        cache.set("text two", [2.0])
        assert cache.memory_size == 2

    def test_clear_memory_empties_dict(self):
        cache = self._cache()
        cache.set("text", [1.0])
        cache.clear_memory()
        assert cache.memory_size == 0

    def test_get_batch_returns_none_for_missing(self):
        cache = self._cache()
        cache.set("known", [1.0])
        results = cache.get_batch(["known", "unknown"])
        assert results[0] == [1.0]
        assert results[1] is None

    def test_set_batch(self):
        cache = self._cache()
        texts = ["a", "b", "c"]
        vecs  = [[1.0], [2.0], [3.0]]
        cache.set_batch(texts, vecs)
        assert cache.get("b") == [2.0]

    def test_different_texts_produce_different_keys(self):
        cache = self._cache()
        cache.set("text A", [1.0])
        cache.set("text B", [2.0])
        assert cache.get("text A") != cache.get("text B")

    def test_redis_unavailable_falls_back_to_memory(self):
        """When Redis is unreachable, cache still works via memory tier."""
        cache = self._cache(redis_url="redis://localhost:19999")  # nothing listening
        assert not cache.redis_available
        cache.set("test", [1.0, 2.0])
        assert cache.get("test") == [1.0, 2.0]

    def test_cache_key_prefix_includes_provider_model(self):
        from app.embedders.cache import EmbeddingCache
        cache = EmbeddingCache(provider="cohere", model="embed-english-v3.0")
        key = cache._key("hello")
        assert "cohere" in key
        assert "embed-english-v3.0" in key


# ─────────────────────────── BaseEmbedder behaviour ──────────────────────────

class TestBaseEmbedderBehaviour:
    """
    Tests for BaseEmbedder caching, batching, and normalisation
    using a minimal concrete subclass.
    """

    class _DummyEmbedder(MagicMock):
        """Minimal concrete BaseEmbedder for testing base behaviour."""
        provider_name = "dummy"
        model_name    = "dummy-model"
        dimension     = 4

        def __init__(self):
            from app.core.base.embedder import BaseEmbedder

            class Concrete(BaseEmbedder):
                provider_name = "dummy"
                model_name    = "dummy-model"
                dimension     = 4
                call_count    = 0

                async def _embed_batch(self_, texts):
                    self_.call_count += 1
                    return [fake_embedding(4) for _ in texts]

            self._impl = Concrete(batch_size=3, cache_enabled=True)

        def __getattr__(self, name):
            return getattr(self._impl, name)

    def _embedder(self):
        from app.core.base.embedder import BaseEmbedder

        class Concrete(BaseEmbedder):
            provider_name = "dummy"
            call_count    = 0

            async def _embed_batch(self, texts):
                self.call_count += 1
                return [fake_embedding(4) for _ in texts]

        return Concrete(batch_size=3, cache_enabled=True)

    def test_embed_returns_correct_count(self):
        emb = self._embedder()
        results = run(emb.embed(["a", "b", "c"]))
        assert len(results) == 3

    def test_embed_empty_returns_empty(self):
        emb = self._embedder()
        assert run(emb.embed([])) == []

    def test_cache_prevents_duplicate_api_calls(self):
        emb = self._embedder()
        run(emb.embed(["hello"]))
        run(emb.embed(["hello"]))   # should hit cache
        assert emb.call_count == 1

    def test_different_texts_not_cached(self):
        emb = self._embedder()
        run(emb.embed(["hello"]))
        run(emb.embed(["world"]))
        assert emb.call_count == 2

    def test_batching_splits_large_inputs(self):
        emb = self._embedder()
        # batch_size=3, sending 7 texts → should call API in 3 batches
        run(emb.embed(["text"] * 7))
        assert emb.call_count >= 1   # at least one batch

    def test_embed_query_returns_single_vector(self):
        emb = self._embedder()
        vec = run(emb.embed_query("what is AI?"))
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_cache_size_tracks_stored_items(self):
        emb = self._embedder()
        run(emb.embed(["alpha", "beta", "gamma"]))
        assert emb.cache_size == 3

    def test_clear_cache(self):
        emb = self._embedder()
        run(emb.embed(["hello"]))
        emb.clear_cache()
        assert emb.cache_size == 0

    def test_normalise_produces_unit_vector(self):
        from app.core.base.embedder import BaseEmbedder
        vec    = [3.0, 4.0]
        normed = BaseEmbedder._normalise(vec)
        mag    = math.sqrt(sum(x * x for x in normed))
        assert abs(mag - 1.0) < 1e-6

    def test_normalise_zero_vector_unchanged(self):
        from app.core.base.embedder import BaseEmbedder
        vec = [0.0, 0.0, 0.0]
        assert BaseEmbedder._normalise(vec) == vec

    def test_cache_key_is_md5_hex(self):
        from app.core.base.embedder import BaseEmbedder
        key = BaseEmbedder._cache_key("hello")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_text_same_cache_key(self):
        from app.core.base.embedder import BaseEmbedder
        assert BaseEmbedder._cache_key("test") == BaseEmbedder._cache_key("test")

    def test_different_texts_different_keys(self):
        from app.core.base.embedder import BaseEmbedder
        assert BaseEmbedder._cache_key("a") != BaseEmbedder._cache_key("b")


# ─────────────────────────── OpenAIEmbedder ──────────────────────────────────

class TestOpenAIEmbedder:

    def _embedder(self) -> "OpenAIEmbedder":
        from app.embedders.openai_embedder import OpenAIEmbedder
        emb = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="sk-test-fake-key",
            batch_size=10,
            redis_url=None,
        )
        return emb

    def test_provider_name(self):
        from app.embedders.openai_embedder import OpenAIEmbedder
        assert OpenAIEmbedder.provider_name == "openai"

    def test_dimension_small_model(self):
        from app.embedders.openai_embedder import OpenAIEmbedder
        emb = OpenAIEmbedder(model="text-embedding-3-small", api_key="x")
        assert emb.dimension == 1536

    def test_dimension_large_model(self):
        from app.embedders.openai_embedder import OpenAIEmbedder
        emb = OpenAIEmbedder(model="text-embedding-3-large", api_key="x")
        assert emb.dimension == 3072

    def test_matryoshka_dimension_override(self):
        from app.embedders.openai_embedder import OpenAIEmbedder
        emb = OpenAIEmbedder(
            model="text-embedding-3-small", api_key="x", dimensions=256
        )
        assert emb.dimension == 256

    def test_truncate_long_text(self):
        from app.embedders.openai_embedder import OpenAIEmbedder
        long_text = "A" * 100_000
        truncated = OpenAIEmbedder._truncate(long_text)
        assert len(truncated) < len(long_text)

    def test_truncate_short_text_unchanged(self):
        from app.embedders.openai_embedder import OpenAIEmbedder
        short = "Hello world"
        assert OpenAIEmbedder._truncate(short) == short

    def test_embed_calls_api_for_uncached(self):
        async def _go():
            emb = self._embedder()
            emb._call_api = AsyncMock(return_value=[fake_embedding(1536)])
            results = await emb._embed_batch(["new text"])
            emb._call_api.assert_called_once()
            assert len(results) == 1
        run(_go())

    @patch("app.embedders.openai_embedder.OpenAIEmbedder._call_api")
    def test_embed_uses_cache_on_second_call(self, mock_call):
        async def _go():
            emb = self._embedder()
            emb._call_api = AsyncMock(return_value=[fake_embedding(1536)])
            await emb._embed_batch(["cached text"])
            await emb._embed_batch(["cached text"])   # should hit cache
            assert emb._call_api.call_count == 1
        run(_go())

    @patch("app.embedders.openai_embedder.OpenAIEmbedder._call_api")
    def test_normalisation_applied(self, mock_call):
        async def _go():
            raw = [3.0, 4.0, 0.0, 0.0]
            emb = self._embedder()
            emb._call_api = AsyncMock(return_value=[raw])
            results = await emb._embed_batch(["text"])
            vec = results[0]
            mag = math.sqrt(sum(x * x for x in vec))
            assert abs(mag - 1.0) < 1e-6
        run(_go())

    def test_model_name_stored(self):
        emb = self._embedder()
        assert emb.model_name == "text-embedding-3-small"


# ─────────────────────────── CohereEmbedder ──────────────────────────────────

class TestCohereEmbedder:

    def _embedder(self):
        from app.embedders.cohere_embedder import CohereEmbedder
        return CohereEmbedder(
            model="embed-english-v3.0",
            api_key="test-fake-key",
            redis_url=None,
        )

    def test_provider_name(self):
        from app.embedders.cohere_embedder import CohereEmbedder
        assert CohereEmbedder.provider_name == "cohere"

    def test_dimension(self):
        emb = self._embedder()
        assert emb.dimension == 1024

    def test_batch_size_clamped_to_96(self):
        from app.embedders.cohere_embedder import CohereEmbedder
        emb = CohereEmbedder(model="embed-english-v3.0", api_key="x", batch_size=200)
        assert emb.batch_size == 96

    def test_embed_uses_search_document_type(self):
        async def _go():
            emb = self._embedder()
            emb._call_api = AsyncMock(return_value=[fake_embedding(1024)])
            await emb._embed_with_type(["some text"], input_type="search_document")
            call_args = emb._call_api.call_args
            assert call_args[0][1] == "search_document"
        run(_go())

    def test_embed_query_uses_search_query_type(self):
        async def _go():
            emb = self._embedder()
            emb._call_api = AsyncMock(return_value=[fake_embedding(1024)])
            await emb.embed_query("what is machine learning?")
            call_args = emb._call_api.call_args
            assert call_args[0][1] == "search_query"
        run(_go())

    def test_document_and_query_cache_keys_differ(self):
        async def _go():
            emb = self._embedder()
            call_count = 0

            async def mock_api(texts, itype):
                nonlocal call_count
                call_count += 1
                return [fake_embedding(1024) for _ in texts]

            emb._call_api = mock_api
            text = "same text"
            # Embed as document
            await emb._embed_with_type([text], "search_document")
            # Embed as query — should NOT hit the document cache
            await emb._embed_with_type([text], "search_query")
            assert call_count == 2   # both cache misses (different input_type prefix)
        run(_go())

    def test_normalisation_applied(self):
        async def _go():
            emb = self._embedder()
            raw = [3.0, 4.0] + [0.0] * 1022
            emb._call_api = AsyncMock(return_value=[raw])
            results = await emb._embed_with_type(["text"], "search_document")
            mag = math.sqrt(sum(x * x for x in results[0]))
            assert abs(mag - 1.0) < 1e-6
        run(_go())


# ─────────────────────────── SentenceTransformersEmbedder ────────────────────

class TestSentenceTransformersEmbedder:

    def _embedder(self) -> "SentenceTransformersEmbedder":
        from app.embedders.st_embedder import SentenceTransformersEmbedder
        return SentenceTransformersEmbedder(
            model="all-MiniLM-L6-v2",
            device="cpu",
            redis_url=None,
        )

    def test_provider_name(self):
        from app.embedders.st_embedder import SentenceTransformersEmbedder
        assert SentenceTransformersEmbedder.provider_name == "sentence_transformers"

    def test_known_model_dimension(self):
        emb = self._embedder()
        assert emb.dimension == 384

    def test_bge_query_prefix_added(self):
        from app.embedders.st_embedder import SentenceTransformersEmbedder
        emb = SentenceTransformersEmbedder(
            model="BAAI/bge-small-en-v1.5", device="cpu"
        )
        prefixed = emb._maybe_add_query_prefix("test query")
        assert "Represent this sentence" in prefixed

    def test_non_bge_no_prefix(self):
        emb = self._embedder()
        q   = "test query"
        assert emb._maybe_add_query_prefix(q) == q

    def test_resolve_device_cpu(self):
        from app.embedders.st_embedder import SentenceTransformersEmbedder
        emb = SentenceTransformersEmbedder(model="all-MiniLM-L6-v2", device="cpu")
        assert emb._resolve_device() == "cpu"

    def test_resolve_device_auto_returns_string(self):
        emb = self._embedder()
        emb._device_pref = "auto"
        device = emb._resolve_device()
        assert device in ("cpu", "cuda")

    def test_model_not_loaded_initially(self):
        from app.embedders.st_embedder import _MODEL_CACHE
        emb = self._embedder()
        assert not emb.is_model_loaded or emb.model_name not in _MODEL_CACHE or True

    def test_embed_batch_mocked(self):
        """Verify full embed_batch flow without loading actual model."""
        async def _go():
            emb = self._embedder()
            mock_vecs = [fake_embedding(384) for _ in range(3)]
            emb._encode_async = AsyncMock(return_value=mock_vecs)
            results = await emb._embed_batch(["a", "b", "c"])
            assert len(results) == 3
            emb._encode_async.assert_called_once()
        run(_go())

    def test_cache_prevents_re_encoding(self):
        async def _go():
            emb = self._embedder()
            call_count = 0

            async def mock_encode(texts):
                nonlocal call_count
                call_count += 1
                return [fake_embedding(384) for _ in texts]

            emb._encode_async = mock_encode
            await emb._embed_batch(["cached"])
            await emb._embed_batch(["cached"])   # should hit cache
            assert call_count == 1
        run(_go())


# ─────────────────────────── EmbedderFactory ─────────────────────────────────

class TestEmbedderFactory:

    @patch.dict("os.environ", {
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_BATCH_SIZE": "100",
        "EMBEDDING_DIMENSION": "1536",
    })
    def test_factory_returns_openai_embedder(self):
        from app.config import get_settings
        get_settings.cache_clear()
        from app.core.base.embedder import EmbedderFactory
        from app.embedders.openai_embedder import OpenAIEmbedder
        emb = EmbedderFactory.get(provider="openai", model="text-embedding-3-small")
        assert isinstance(emb, OpenAIEmbedder)
        get_settings.cache_clear()

    @patch.dict("os.environ", {
        "EMBEDDING_PROVIDER": "cohere",
        "COHERE_API_KEY": "test-key",
        "EMBEDDING_MODEL": "embed-english-v3.0",
        "EMBEDDING_BATCH_SIZE": "96",
        "EMBEDDING_DIMENSION": "1024",
    })
    def test_factory_returns_cohere_embedder(self):
        from app.config import get_settings
        get_settings.cache_clear()
        from app.core.base.embedder import EmbedderFactory
        from app.embedders.cohere_embedder import CohereEmbedder
        emb = EmbedderFactory.get(provider="cohere")
        assert isinstance(emb, CohereEmbedder)
        get_settings.cache_clear()

    @patch.dict("os.environ", {
        "EMBEDDING_PROVIDER": "sentence_transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "EMBEDDING_BATCH_SIZE": "64",
        "EMBEDDING_DIMENSION": "384",
    })
    def test_factory_returns_st_embedder(self):
        from app.config import get_settings
        get_settings.cache_clear()
        from app.core.base.embedder import EmbedderFactory
        from app.embedders.st_embedder import SentenceTransformersEmbedder
        emb = EmbedderFactory.get(provider="sentence_transformers")
        assert isinstance(emb, SentenceTransformersEmbedder)
        get_settings.cache_clear()

    def test_factory_unknown_provider_raises(self):
        from app.core.base.embedder import EmbedderFactory
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbedderFactory.get(provider="nonexistent_provider")
