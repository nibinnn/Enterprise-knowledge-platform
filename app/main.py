"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

Routers are registered here as they are built on Days 21-22.
For now, only the /health and /ready endpoints are live.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.database import check_db_connection, close_db, init_db

settings = get_settings()

# ── Structured logging setup ──────────────────────────────────────────────────

structlog.configure(
    # processors=[
    #     structlog.stdlib.add_log_level,
    #     structlog.stdlib.add_logger_name,
    #     structlog.processors.TimeStamper(fmt="iso"),
    #     structlog.dev.ConsoleRenderer() 
    #     if settings.app_debug
    #     else structlog.processors.JSONRenderer(),
    # ],
    # wrapper_class=structlog.stdlib.BoundLogger,
    # logger_factory=structlog.PrintLoggerFactory(),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.app_debug
        else structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = structlog.get_logger()


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info(
        "Starting EKIP",
        env=settings.app_env.value,
        debug=settings.app_debug,
    )

    # Startup
    if not settings.is_production:
        await init_db()          # creates tables in dev; use Alembic in prod
        logger.info("Database tables ensured.")

    logger.info("EKIP ready.", port=settings.app_port)

    yield   # ← app is running

    # Shutdown
    await close_db()
    logger.info("EKIP shutdown complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description=(
        "An AI Agent that can understand, search, analyze, "
        "and answer questions from thousands of enterprise documents."
    ),
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# CORS
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Error handlers ────────────────────────────────────────────────────────────
from app.api.errors import register_handlers
register_handlers(app)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routes import auth, documents, search, ask, feedback, admin

PREFIX = "/api/v1"
app.include_router(auth.router,      prefix=PREFIX)
app.include_router(documents.router, prefix=PREFIX)
app.include_router(search.router,    prefix=PREFIX)
app.include_router(ask.router,       prefix=PREFIX)
app.include_router(feedback.router,  prefix=PREFIX)
app.include_router(admin.router,     prefix=PREFIX)


# ── Ops endpoints ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> Dict[str, Any]:
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "ok", "service": "ekip"}


@app.get("/ready", tags=["ops"])
async def ready() -> JSONResponse:
    """
    Readiness probe — checks all downstream dependencies.
    Returns 200 only if Postgres (and eventually Qdrant) are reachable.
    """
    checks: Dict[str, Any] = {}
    all_ok = True

    # Postgres
    db_ok = await check_db_connection()
    checks["postgres"] = "ok" if db_ok else "unreachable"
    if not db_ok:
        all_ok = False

    # Qdrant (added on Day 7)
    # qdrant_ok = await check_qdrant_connection()
    # checks["qdrant"] = "ok" if qdrant_ok else "unreachable"

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )
