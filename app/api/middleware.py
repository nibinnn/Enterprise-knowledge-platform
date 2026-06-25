"""
app/api/middleware.py
─────────────────────────────────────────────────────────────────────────────
FastAPI middleware stack:

  RequestIDMiddleware   — injects X-Request-ID header; correlates logs with traces
  MetricsMiddleware     — records HTTP request counts + latency in Prometheus
  LoggingMiddleware     — structured per-request log line with timing and trace id

All three are registered together via `register_middleware(app)`.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import time
import uuid
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths excluded from metrics (noisy, low-value)
_EXCLUDED_PATHS = {"/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}


# ─────────────────────────── Request ID ──────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns every request a unique ID.
    Reads X-Request-ID from the incoming request if present (for tracing
    across services); generates a new UUID otherwise.
    Echoes the id back in the response header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = (
            request.headers.get("x-request-id")
            or str(uuid.uuid4())
        )
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ─────────────────────────── Prometheus metrics ───────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """Records ekip_http_requests_total and ekip_http_request_duration_seconds."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Don't record metrics for health/metrics endpoints themselves
        if path in _EXCLUDED_PATHS:
            return await call_next(request)

        # Normalise dynamic path segments (/documents/{id} → /documents/{id})
        # FastAPI already provides the route template — use it if available
        route = getattr(request, "route", None)
        path_template = getattr(route, "path", path) if route else path

        start  = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status   = response.status_code
            return response
        except Exception:
            raise
        finally:
            duration = time.perf_counter() - start
            try:
                from app.core.metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION
                HTTP_REQUESTS_TOTAL.labels(
                    method=request.method,
                    path=path_template,
                    status_code=str(status),
                ).inc()
                HTTP_REQUEST_DURATION.labels(
                    method=request.method,
                    path=path_template,
                ).observe(duration)
            except Exception as exc:
                logger.debug("Metrics recording failed: %s", exc)


# ─────────────────────────── Structured access log ───────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Emits one structured log line per request:
      method=GET path=/api/v1/ask status=200 duration_ms=342 request_id=... trace_id=...
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        start      = time.perf_counter()
        request_id = getattr(request.state, "request_id", "-")

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        from app.core.tracing import get_trace_id
        logger.info(
            "method=%s path=%s status=%d duration_ms=%.1f request_id=%s trace_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
            get_trace_id() or "-",
        )
        return response


# ─────────────────────────── Registration helper ─────────────────────────────

def register_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI app.
    Order matters: last added = outermost (runs first on request, last on response).
    """
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)   # outermost — assigns ID first
