"""
app/api/auth.py
─────────────────────────────────────────────────────────────────────────────
JWT + API key authentication utilities.

Two auth methods supported:
  1. JWT Bearer token  — for interactive users (short-lived, 60 min default)
  2. API Key           — for programmatic / service-to-service access
                         (long-lived, stored hashed in DB)

Both methods produce the same CurrentUser object injected into routes.

Security notes:
  - JWT secret is read from SECRET_KEY env var — change before prod
  - API keys are stored as bcrypt hashes; the plaintext is shown once at creation
  - Passwords hashed with passlib bcrypt (cost factor 12)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.api.schemas.auth import CurrentUser
from app.config import get_settings

settings = get_settings()

# ── Password hashing ──────────────────────────────────────────────────────────
# Using sha256_crypt for dev compatibility (bcrypt 4.x removed __about__
# which passlib depends on). Switch to argon2 or bcrypt<4 in production.
_pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ── API key utilities ─────────────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.
    Returns: (plaintext_key, hashed_key)
    Store the hash; show the plaintext exactly once.
    """
    plaintext = f"ekip-{secrets.token_urlsafe(32)}"
    hashed    = hashlib.sha256(plaintext.encode()).hexdigest()
    return plaintext, hashed


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


# ── JWT utilities ─────────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    username: str,
    scopes: list[str],
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {
        "sub":      user_id,
        "username": username,
        "scopes":   scopes,
        "exp":      expire,
        "iat":      datetime.now(timezone.utc),
        "type":     "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT.
    Raises HTTP 401 on any validation failure.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        if payload.get("type") != "access":
            raise credentials_exception
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise credentials_exception
        return payload
    except JWTError:
        raise credentials_exception


def token_to_current_user(payload: dict) -> CurrentUser:
    return CurrentUser(
        user_id=payload["sub"],
        username=payload.get("username", payload["sub"]),
        scopes=payload.get("scopes", []),
        auth_method="jwt",
    )


# ── Scope checking ────────────────────────────────────────────────────────────

def require_scope(user: CurrentUser, scope: str) -> None:
    """Raise HTTP 403 if the user does not have the required scope."""
    if scope not in user.scopes and "admin" not in user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Scope '{scope}' required.",
        )
