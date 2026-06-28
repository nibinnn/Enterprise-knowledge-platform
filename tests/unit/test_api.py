"""
tests/unit/test_api.py
─────────────────────────────────────────────────────────────────────────────
API unit tests using FastAPI's TestClient.
DB and service calls are mocked so no real DB/Qdrant is needed.
Run with:  pytest tests/unit/test_api.py -v
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── App fixture ───────────────────────────────────────────────────────────────
# Pre-import DB modules so SQLAlchemy engine is created (no real connection yet)
import app.db.database as _db_module          # noqa — resolves patch targets
import app.db.models                          # noqa

from app.db.database import get_db as _get_db
from app.main import app as _app


async def _fake_db():
    """In-memory mock DB session injected into all routes during tests."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value=None),
        scalar_one=MagicMock(return_value=0),
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
    ))
    db.add     = MagicMock()
    db.flush   = AsyncMock()
    db.commit  = AsyncMock()
    db.rollback = AsyncMock()
    db.close   = AsyncMock()
    yield db


@pytest.fixture(scope="module")
def client():
    """TestClient with DB dependency overridden and lifespan mocked."""
    _app.dependency_overrides[_get_db] = _fake_db

    with (
        patch("app.main.init_db",  AsyncMock()),
        patch("app.main.close_db", AsyncMock()),
        patch("app.db.database.check_db_connection", AsyncMock(return_value=True)),
    ):
        with TestClient(_app, raise_server_exceptions=False) as c:
            yield c

    _app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Obtain a real JWT token from the /auth/token endpoint."""
    resp = client.post("/api/v1/auth/token", json={"username": "admin", "password": "changeme"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ─────────────────────────── Ops endpoints ────────────────────────────────────

class TestOpsEndpoints:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_returns_200_or_503(self, client):
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)
        assert "status" in resp.json()

    def test_docs_available_in_dev(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


# ─────────────────────────── Auth routes ─────────────────────────────────────

class TestAuthRoutes:

    def test_login_correct_credentials(self, client):
        resp = client.post("/api/v1/auth/token",
                           json={"username": "admin", "password": "changeme"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    def test_login_wrong_password(self, client):
        resp = client.post("/api/v1/auth/token",
                           json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_wrong_username(self, client):
        resp = client.post("/api/v1/auth/token",
                           json={"username": "hacker", "password": "changeme"})
        assert resp.status_code == 401

    def test_whoami_authenticated(self, client, auth_headers):
        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["username"] == "admin"
        assert "admin" in data["scopes"]

    def test_whoami_unauthenticated(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_whoami_invalid_token(self, client):
        resp = client.get("/api/v1/auth/me",
                          headers={"Authorization": "Bearer totally.fake.token"})
        assert resp.status_code == 401


# ─────────────────────────── Document routes ─────────────────────────────────

class TestDocumentRoutes:

    def _mock_db(self):
        db = AsyncMock()
        db.execute = AsyncMock()
        db.flush   = AsyncMock()
        db.add     = MagicMock()
        return db

    def test_list_documents_requires_auth(self, client):
        resp = client.get("/api/v1/documents/")
        assert resp.status_code == 401

    def test_upload_requires_auth(self, client):
        resp = client.post("/api/v1/documents/upload",
                           files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")})
        assert resp.status_code == 401

    def test_upload_unsupported_extension(self, client, auth_headers):
        with (
            patch("app.api.routes.documents.get_db",
                  return_value=AsyncMock()),
            patch("app.api.routes.documents.get_ingestion_service",
                  return_value=MagicMock()),
        ):
            resp = client.post(
                "/api/v1/documents/upload",
                files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp.status_code in (415, 422, 500)

    def test_get_document_not_found(self, client, auth_headers):
        fake_id = str(uuid.uuid4())
        with patch("app.api.routes.documents.get_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_get_db.return_value = mock_db

            resp = client.get(f"/api/v1/documents/{fake_id}", headers=auth_headers)
        assert resp.status_code in (404, 500)

    def test_delete_document_not_found(self, client, auth_headers):
        fake_id = str(uuid.uuid4())
        with (
            patch("app.api.routes.documents.get_db", return_value=AsyncMock()),
            patch("app.api.routes.documents.get_ingestion_service") as mock_svc_dep,
        ):
            mock_svc = AsyncMock()
            mock_svc.delete_document = AsyncMock(return_value=False)
            mock_svc_dep.return_value = mock_svc
            resp = client.delete(f"/api/v1/documents/{fake_id}", headers=auth_headers)
        assert resp.status_code in (404, 500)


# ─────────────────────────── Search routes ───────────────────────────────────

class TestSearchRoutes:

    def test_search_requires_auth(self, client):
        resp = client.post("/api/v1/search/", json={"query": "test"})
        assert resp.status_code == 401

    def test_search_returns_empty_results_stub(self, client, auth_headers):
        from app.api.schemas.search import SearchResponse
        from app.services.search import SearchService
        from app.api.dependencies import get_search_service
        async def _mock_svc():
            svc = MagicMock(spec=SearchService)
            svc.search = AsyncMock(return_value=SearchResponse(
                query="machine learning", results=[], total_results=0, mode="hybrid", latency_ms=1.0
            ))
            return svc
        _app.dependency_overrides[get_search_service] = _mock_svc
        resp = client.post("/api/v1/search/",
                           json={"query": "machine learning"},
                           headers=auth_headers)
        _app.dependency_overrides.pop(get_search_service, None)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["query"] == "machine learning"

    def test_search_empty_query_rejected(self, client, auth_headers):
        with (
            patch("app.api.routes.search.get_db", return_value=AsyncMock()),
            patch("app.api.routes.search.get_search_service", return_value=AsyncMock()),
        ):
            resp = client.post("/api/v1/search/",
                               json={"query": ""},
                               headers=auth_headers)
        assert resp.status_code == 422

    def test_search_top_k_validation(self, client, auth_headers):
        with (
            patch("app.api.routes.search.get_db", return_value=AsyncMock()),
            patch("app.api.routes.search.get_search_service", return_value=AsyncMock()),
        ):
            resp = client.post("/api/v1/search/",
                               json={"query": "test", "top_k": 999},
                               headers=auth_headers)
        assert resp.status_code == 422


# ─────────────────────────── Ask routes ──────────────────────────────────────

class TestAskRoutes:

    def test_ask_requires_auth(self, client):
        resp = client.post("/api/v1/ask/", json={"question": "What is AI?"})
        assert resp.status_code == 401

    def test_ask_returns_stub_answer(self, client, auth_headers):
        from app.api.schemas.ask import AskResponse
        from app.services.ask import AskService
        from app.api.dependencies import get_ask_service
        async def _mock_svc():
            svc = MagicMock(spec=AskService)
            svc.ask = AsyncMock(return_value=AskResponse(
                answer_id=str(uuid.uuid4()),
                question="What is AI?",
                answer="Artificial Intelligence is...",
                model_used="claude-sonnet-4-6",
                created_at=datetime.utcnow(),
            ))
            return svc
        _app.dependency_overrides[get_ask_service] = _mock_svc
        resp = client.post("/api/v1/ask/",
                           json={"question": "What is AI?"},
                           headers=auth_headers)
        _app.dependency_overrides.pop(get_ask_service, None)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "answer" in body["data"]

    def test_ask_empty_question_rejected(self, client, auth_headers):
        with (
            patch("app.api.routes.ask.get_db", return_value=AsyncMock()),
            patch("app.api.routes.ask.get_ask_service", return_value=AsyncMock()),
        ):
            resp = client.post("/api/v1/ask/",
                               json={"question": ""},
                               headers=auth_headers)
        assert resp.status_code == 422


# ─────────────────────────── Feedback routes ─────────────────────────────────

class TestFeedbackRoutes:

    def test_feedback_requires_auth(self, client):
        resp = client.post("/api/v1/feedback",
                           json={"answer_id": "x", "rating": 5})
        assert resp.status_code == 401

    def test_feedback_rating_out_of_range(self, client, auth_headers):
        with patch("app.api.routes.feedback.get_db", return_value=AsyncMock()):
            resp = client.post("/api/v1/feedback",
                               json={"answer_id": "x", "rating": 10},
                               headers=auth_headers)
        assert resp.status_code == 422

    def test_list_eval_runs_empty(self, client, auth_headers):
        with patch("app.api.routes.feedback.get_db") as mock_dep:
            mock_db = AsyncMock()
            mock_count = MagicMock(); mock_count.scalar_one.return_value = 0
            mock_rows  = MagicMock(); mock_rows.scalars.return_value.all.return_value = []
            mock_db.execute = AsyncMock(side_effect=[mock_count, mock_rows])
            mock_dep.return_value = mock_db
            resp = client.get("/api/v1/eval/runs", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"] == []


# ─────────────────────────── Schema validation ───────────────────────────────

class TestSchemaValidation:

    def test_search_request_mode_default(self):
        from app.api.schemas.search import SearchRequest
        r = SearchRequest(query="test")
        assert r.mode == "hybrid"
        assert r.top_k == 10

    def test_ask_request_defaults(self):
        from app.api.schemas.ask import AskRequest
        r = AskRequest(question="What is X?")
        assert r.mode == "rag"
        assert not r.stream
        assert r.top_k == 20
        assert r.rerank_top_n == 5

    def test_feedback_rating_bounds(self):
        from app.api.schemas.feedback import FeedbackRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FeedbackRequest(answer_id="x", rating=0)
        with pytest.raises(ValidationError):
            FeedbackRequest(answer_id="x", rating=6)
        FeedbackRequest(answer_id="x", rating=1)
        FeedbackRequest(answer_id="x", rating=5)

    def test_pagination_offset(self):
        from app.api.schemas.common import PaginationParams
        p = PaginationParams(page=3, page_size=10)
        assert p.offset == 20

    def test_document_filter_valid_status(self):
        from app.api.schemas.documents import DocumentFilterParams
        f = DocumentFilterParams(status="indexed")
        assert f.status == "indexed"

    def test_document_filter_invalid_status(self):
        from app.api.schemas.documents import DocumentFilterParams
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentFilterParams(status="nonexistent")

    def test_auth_token_response(self):
        from app.api.schemas.auth import TokenResponse
        r = TokenResponse(access_token="abc", expires_in=3600)
        assert r.token_type == "bearer"
