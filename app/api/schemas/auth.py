"""app/api/schemas/auth.py — authentication schemas."""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int       # seconds


class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default_factory=lambda: ["read", "write"])
    expires_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    key_id: str
    name: str
    api_key: str          # shown ONCE at creation, never again
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None


class APIKeyOut(BaseModel):
    """Safe version — api_key masked after initial creation."""
    key_id: str
    name: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class CurrentUser(BaseModel):
    """Injected into route handlers via Depends(get_current_user)."""
    user_id: str
    username: str
    scopes: List[str] = Field(default_factory=list)
    auth_method: str = "jwt"    # jwt | api_key
