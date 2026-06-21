"""app/api/routes/auth.py — authentication endpoints."""
from __future__ import annotations
import uuid
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import (
    create_access_token, generate_api_key,
    hash_password, verify_password,
)
from app.api.dependencies import get_current_user, get_db
from app.api.schemas.auth import (
    APIKeyCreateRequest, APIKeyOut, APIKeyResponse,
    CurrentUser, TokenRequest, TokenResponse,
)
from app.api.schemas.common import APIResponse
from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()

# ---------------------------------------------------------------------------
# NOTE: For the MVP we use a single hardcoded admin user defined in .env.
# Replace with a proper User DB table before going to production.
# ---------------------------------------------------------------------------
_ADMIN_USERNAME = "admin"
_ADMIN_PASSWORD_HASH: list = []   # lazy-init to avoid bcrypt import-time check


def _admin_hash() -> str:
    """Compute the admin password hash once on first login (lazy init)."""
    if not _ADMIN_PASSWORD_HASH:
        _ADMIN_PASSWORD_HASH.append(hash_password("changeme"))
    return _ADMIN_PASSWORD_HASH[0]


@router.post("/token", response_model=TokenResponse, summary="Obtain JWT access token")
async def login(body: TokenRequest):
    """
    Exchange username + password for a short-lived JWT Bearer token.
    MVP: single admin user. Production: query users table.
    """
    valid = (
        body.username == _ADMIN_USERNAME
        and verify_password(body.password, _admin_hash())
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        user_id=_ADMIN_USERNAME,
        username=_ADMIN_USERNAME,
        scopes=["read", "write", "admin"],
    )
    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/api-keys",
    response_model=APIResponse[APIKeyResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key",
)
async def create_api_key(
    body: APIKeyCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a new API key. The plaintext key is shown ONCE — store it safely.
    Requires 'admin' scope.
    """
    from app.api.auth import require_scope
    require_scope(current_user, "admin")

    plaintext, key_hash = generate_api_key()
    key_id     = str(uuid.uuid4())
    created_at = datetime.utcnow()
    expires_at = (
        created_at + timedelta(days=body.expires_days)
        if body.expires_days else None
    )

    # TODO (Day 9 follow-up): persist to APIKeyModel table in DB
    # db.add(APIKeyModel(id=key_id, name=body.name, key_hash=key_hash,
    #                    scopes=body.scopes, expires_at=expires_at,
    #                    owner_id=current_user.user_id))

    return APIResponse(
        data=APIKeyResponse(
            key_id=key_id,
            name=body.name,
            api_key=plaintext,
            scopes=body.scopes,
            created_at=created_at,
            expires_at=expires_at,
        )
    )


@router.get(
    "/api-keys",
    response_model=APIResponse[List[APIKeyOut]],
    summary="List API keys (masked)",
)
async def list_api_keys(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys owned by the current user (plaintext never returned)."""
    from app.api.auth import require_scope
    require_scope(current_user, "admin")
    # TODO: query APIKeyModel from DB
    return APIResponse(data=[])


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently revoke an API key."""
    from app.api.auth import require_scope
    require_scope(current_user, "admin")
    # TODO: delete from APIKeyModel in DB
    return None


@router.get("/me", response_model=APIResponse[CurrentUser])
async def whoami(current_user: CurrentUser = Depends(get_current_user)):
    """Return the currently authenticated user's identity."""
    return APIResponse(data=current_user)
