"""
app/api/errors.py
─────────────────────────────────────────────────────────────────────────────
Global exception handlers. Registered once in main.py via register_handlers().

Every unhandled exception is converted to a consistent ErrorResponse JSON.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def _error_body(code: str, message: str, field: str | None = None) -> dict:
    return {
        "success": False,
        "error": {"code": code, "message": message, "field": field},
        "meta": {"request_id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat()},
    }


def register_handlers(app: FastAPI) -> None:

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field   = ".".join(str(l) for l in first.get("loc", [])[1:]) or None
        message = first.get("msg", "Validation error")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body("VALIDATION_ERROR", message, field),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body("BAD_REQUEST", str(exc)),
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=_error_body("FORBIDDEN", str(exc)),
        )

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_body("NOT_FOUND", str(exc)),
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("INTERNAL_ERROR", "An unexpected error occurred."),
        )
