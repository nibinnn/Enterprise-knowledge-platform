"""
tests/conftest.py
Sets environment variables before any app module is imported so the
SQLAlchemy engines use SQLite (no Postgres / psycopg2 needed in unit tests).
"""
import os

# Must be set BEFORE app modules are imported
os.environ.setdefault("DATABASE_URL",       "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("OPENAI_API_KEY",     "sk-test-key")
os.environ.setdefault("ANTHROPIC_API_KEY",  "sk-ant-test-key")
os.environ.setdefault("COHERE_API_KEY",     "test-cohere-key")
os.environ.setdefault("SECRET_KEY",         "test-secret-key-32-chars-minimum!")
os.environ.setdefault("APP_ENV",            "development")
os.environ.setdefault("QDRANT_HOST",        "localhost")
os.environ.setdefault("REDIS_URL",          "redis://localhost:6379/0")

# Clear the settings cache so it picks up the overrides
from app.config import get_settings
get_settings.cache_clear()
