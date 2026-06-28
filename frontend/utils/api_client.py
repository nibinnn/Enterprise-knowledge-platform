"""
frontend/utils/api_client.py
HTTP client for the FastAPI backend. Streamlit is sync, so we use requests.
All methods return (data, error) tuples — caller decides how to surface errors.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
TIMEOUT  = 60   # seconds — LLM calls can be slow


class APIClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._token:   Optional[str] = None
        self._session = requests.Session()

    # ── Auth ─────────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        resp = self._post("/auth/token", json={"username": username, "password": password}, auth=False)
        if resp.ok:
            self._token = resp.json()["access_token"]
            self._session.headers.update({"Authorization": f"Bearer {self._token}"})
            return True, ""
        return False, resp.json().get("error", {}).get("message", "Login failed")

    def is_authenticated(self) -> bool:
        return self._token is not None

    def logout(self) -> None:
        self._token = None
        self._session.headers.pop("Authorization", None)

    # ── Documents ─────────────────────────────────────────────────────────────

    def upload_document(
        self,
        file_bytes:   bytes,
        filename:     str,
        title:        str = "",
        department:   str = "",
        doc_category: str = "",
        tags:         str = "",
    ) -> Tuple[Optional[Dict], Optional[str]]:
        data = {}
        if title:        data["title"]        = title
        if department:   data["department"]   = department
        if doc_category: data["doc_category"] = doc_category
        if tags:         data["tags"]         = tags

        resp = self._session.post(
            f"{self.base_url}/documents/upload",
            files={"file": (filename, file_bytes)},
            data=data,
            timeout=TIMEOUT,
        )
        return self._unwrap(resp)

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        status: str = "",
        search: str = "",
    ) -> Tuple[Optional[Dict], Optional[str]]:
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if status: params["status"] = status
        if search: params["search"] = search
        resp = self._session.get(f"{self.base_url}/documents/", params=params, timeout=TIMEOUT)
        return self._unwrap(resp)

    def get_document_status(self, doc_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        resp = self._session.get(f"{self.base_url}/documents/{doc_id}/status", timeout=TIMEOUT)
        return self._unwrap(resp)

    def delete_document(self, doc_id: str) -> Tuple[bool, Optional[str]]:
        resp = self._session.delete(f"{self.base_url}/documents/{doc_id}", timeout=TIMEOUT)
        if resp.status_code == 204:
            return True, None
        return False, self._error_msg(resp)

    def reindex_document(self, doc_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        resp = self._session.post(f"{self.base_url}/documents/{doc_id}/reindex", timeout=TIMEOUT)
        return self._unwrap(resp)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:        str,
        mode:         str = "hybrid",
        top_k:        int = 10,
        department:   str = "",
        doc_category: str = "",
    ) -> Tuple[Optional[Dict], Optional[str]]:
        filters: Dict[str, Any] = {}
        if department:   filters["department"]   = department
        if doc_category: filters["doc_category"] = doc_category

        payload: Dict[str, Any] = {"query": query, "mode": mode, "top_k": top_k}
        if filters: payload["filters"] = filters

        resp = self._session.post(f"{self.base_url}/search/", json=payload, timeout=TIMEOUT)
        return self._unwrap(resp)

    # ── Ask ───────────────────────────────────────────────────────────────────

    def ask(
        self,
        question:     str,
        mode:         str = "rag",
        top_k:        int = 20,
        rerank_top_n: int = 5,
        department:   str = "",
        doc_category: str = "",
    ) -> Tuple[Optional[Dict], Optional[str]]:
        filters: Dict[str, Any] = {}
        if department:   filters["department"]   = department
        if doc_category: filters["doc_category"] = doc_category

        payload: Dict[str, Any] = {
            "question":    question,
            "mode":        mode,
            "top_k":       top_k,
            "rerank_top_n": rerank_top_n,
            "include_sources": True,
        }
        if filters: payload["filters"] = filters

        endpoint = "/ask/agent" if mode == "agent" else "/ask/"
        resp     = self._session.post(f"{self.base_url}{endpoint}", json=payload, timeout=TIMEOUT)
        return self._unwrap(resp)

    # ── Feedback ──────────────────────────────────────────────────────────────

    def submit_feedback(
        self, answer_id: str, rating: int, correction: str = ""
    ) -> Tuple[Optional[Dict], Optional[str]]:
        payload: Dict[str, Any] = {"answer_id": answer_id, "rating": rating}
        if correction: payload["correction"] = correction
        resp = self._session.post(f"{self.base_url}/feedback", json=payload, timeout=TIMEOUT)
        return self._unwrap(resp)

    # ── Admin ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Tuple[Optional[Dict], Optional[str]]:
        resp = self._session.get(f"{self.base_url}/admin/stats", timeout=TIMEOUT)
        return self._unwrap(resp)

    def get_health(self) -> Tuple[Optional[Dict], Optional[str]]:
        resp = self._session.get(f"{self.base_url}/admin/health/detailed", timeout=TIMEOUT)
        return self._unwrap(resp)

    def list_eval_runs(self) -> Tuple[Optional[Dict], Optional[str]]:
        resp = self._session.get(f"{self.base_url}/eval/runs", timeout=TIMEOUT)
        return self._unwrap(resp)

    def start_eval_run(self, run_name: str, dataset_name: str) -> Tuple[Optional[Dict], Optional[str]]:
        payload = {"run_name": run_name, "dataset_name": dataset_name, "questions": []}
        resp    = self._session.post(f"{self.base_url}/eval/runs", json=payload, timeout=TIMEOUT)
        return self._unwrap(resp)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _post(self, path: str, json: dict, auth: bool = True) -> requests.Response:
        headers = self._session.headers.copy()
        if not auth:
            headers.pop("Authorization", None)
        return self._session.post(
            f"{self.base_url}{path}", json=json, headers=headers, timeout=TIMEOUT
        )

    @staticmethod
    def _unwrap(resp: requests.Response) -> Tuple[Optional[Dict], Optional[str]]:
        if resp.ok:
            body = resp.json()
            return body.get("data", body), None
        return None, APIClient._error_msg(resp)

    @staticmethod
    def _error_msg(resp: requests.Response) -> str:
        try:
            body = resp.json()
            return body.get("error", {}).get("message", resp.text)
        except Exception:
            return resp.text or f"HTTP {resp.status_code}"
