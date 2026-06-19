"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration module using pydantic-settings.
All settings are read from environment variables (or .env file).
Call `get_settings()` anywhere in the codebase — it is cached after first call.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────── Enums (used in Settings) ────────────────────────

class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class SearchMode(str, Enum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ─────────────────────────── Settings ────────────────────────────────────────

class Settings(BaseSettings):
    """
    All platform settings, validated and type-checked at startup.
    Group into nested sub-classes for clarity at the cost of one extra
    object — a deliberate trade-off for a project this size.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # ignore unknown env vars gracefully
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.DEVELOPMENT
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True
    secret_key: str = "change-me"
    log_level: LogLevel = LogLevel.INFO

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PRODUCTION

    # ── Database ─────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ekip"
    postgres_user: str = "ekip_user"
    postgres_password: str = "ekip_pass"
    database_url: str = "postgresql+asyncpg://ekip_user:ekip_pass@localhost:5432/ekip"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    @property
    def sync_database_url(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return self.database_url.replace("postgresql+asyncpg", "postgresql+psycopg2")

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── Vector Database ───────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "ekip_chunks"
    qdrant_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 100
    embedding_cache_ttl: int = 86400

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1

    # ── API Keys ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    context_max_tokens: int = 8000
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Search ────────────────────────────────────────────────────────────────
    search_mode: SearchMode = SearchMode.HYBRID
    hybrid_dense_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3

    @field_validator("hybrid_dense_weight", "hybrid_keyword_weight")
    @classmethod
    def weights_must_be_between_0_and_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Search weights must be between 0.0 and 1.0")
        return v

    # ── Observability ─────────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # ── File Storage ──────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50
    allowed_extensions: str = "pdf,docx,txt,md,html"

    @property
    def allowed_extension_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # ── Auth ──────────────────────────────────────────────────────────────────
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    The cache means .env is read exactly once per process.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()


# ── Module-level convenience (use sparingly — prefer get_settings()) ─────────
settings = get_settings()
