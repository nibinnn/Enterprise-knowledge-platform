"""
app/api/dependencies.py
─────────────────────────────────────────────────────────────────────────────
All FastAPI dependency functions used across routes.

Import pattern in routes:
    from app.api.dependencies import get_db, get_current_user, optional_auth

DB session  →  get_db()
Auth        →  get_current_user()   (required)
              optional_auth()       (returns None if no token provided)
Services    →  get_ingestion_service(), get_search_service(), get_ask_service()
               (stubs now → swap real implementations in Blocks 3-4)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.api.auth import decode_access_token, hash_api_key, token_to_current_user
from app.api.schemas.auth import CurrentUser
from app.config import get_settings
from app.db.database import AsyncSession, get_db          # re-export for routes

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Security schemes ──────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Auth dependencies ─────────────────────────────────────────────────────────

async def get_current_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    api_key: Optional[str]                         = Security(_api_key_header),
    db: AsyncSession                               = Depends(get_db),
) -> CurrentUser:
    """
    Resolve caller identity from JWT Bearer token or X-API-Key header.
    Raises HTTP 401 if neither is provided or both are invalid.
    """
    # ── JWT path ──────────────────────────────────────────────────────────────
    if bearer and bearer.credentials:
        payload = decode_access_token(bearer.credentials)
        return token_to_current_user(payload)

    # ── API key path ──────────────────────────────────────────────────────────
    if api_key:
        user = await _resolve_api_key(api_key, db)
        if user:
            return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def optional_auth(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    api_key: Optional[str]                         = Security(_api_key_header),
    db: AsyncSession                               = Depends(get_db),
) -> Optional[CurrentUser]:
    """
    Like get_current_user but returns None instead of raising 401.
    Use on endpoints that work for both authenticated and anonymous callers.
    """
    try:
        return await get_current_user(bearer, api_key, db)
    except HTTPException:
        return None


async def _resolve_api_key(api_key: str, db: AsyncSession) -> Optional[CurrentUser]:
    """
    Look up an API key in the database.
    Returns CurrentUser on success, None if not found or expired.

    Stub: always returns None until the APIKey DB table is wired up.
    Replace this body with a real DB lookup in Day 9.
    """
    # TODO (Day 9): implement DB lookup
    # key_hash = hash_api_key(api_key)
    # row = await db.execute(select(APIKeyModel).where(APIKeyModel.key_hash == key_hash))
    # ...
    return None


# ── Service stubs — swapped for real implementations in Blocks 3-4 ────────────

def get_ingestion_service():
    """
    Returns the document ingestion service.
    Stub until Celery workers are wired up on Day 19.
    """
    from app.services.ingestion import IngestionService
    return IngestionService()


def get_search_service():
    """
    Returns the search service.
    Stub until Qdrant vector store is built on Day 14.
    """
    from app.services.search import SearchService
    return SearchService()


def get_ask_service():
    """
    Returns the RAG / ask service.
    Stub until RAG pipeline is built on Day 16.
    """
    from app.services.ask import AskService
    return AskService()


def get_agent_service():
    """
    Returns the agent orchestration service.
    Stub until agent layer is built in Block 2.
    """
    from app.services.agent import AgentService
    return AgentService()
