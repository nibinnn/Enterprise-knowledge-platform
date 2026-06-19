"""
app/db/database.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy async engine, session factory, and dependency injection helper.

Usage in FastAPI routes:
    async def route(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(DocumentModel))

Usage in Celery workers (sync):
    with get_sync_session() as db:
        db.execute(...)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()


# ── Async engine (FastAPI) ────────────────────────────────────────────────────

async_engine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.app_debug,          # log SQL in dev
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,           # keep objects accessible after commit
    autoflush=False,
)


# ── Sync engine (Alembic / Celery workers) ────────────────────────────────────

sync_engine = create_engine(
    settings.sync_database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.app_debug,
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


# ── Declarative base (all ORM models inherit from this) ───────────────────────

class Base(DeclarativeBase):
    pass


# ── Dependency injection (FastAPI) ────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency. Yields an async DB session,
    commits on success, rolls back on any exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Context manager (Celery workers / scripts) ────────────────────────────────

@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Synchronous session for Celery workers and CLI scripts."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Startup / shutdown ────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Create all tables on startup (dev only).
    In production, use Alembic migrations instead.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose connection pool on app shutdown."""
    await async_engine.dispose()


async def check_db_connection() -> bool:
    """Health-check: returns True if Postgres is reachable."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
