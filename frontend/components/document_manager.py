"""
frontend/components/document_manager.py
Upload documents, list all documents, poll ingestion status, delete.
"""
from __future__ import annotations

import time

import streamlit as st

from frontend.utils.api_client import APIClient

_STATUS_COLOUR = {
    "pending":    "🟡",
    "processing": "🔵",
    "indexed":    "🟢",
    "failed":     "🔴",
    "archived":   "⚫",
}


def render_document_manager(client: APIClient) -> None:
    tab_upload, tab_list = st.tabs(["📤 Upload", "📋 All Documents"])

    with tab_upload:
        _render_upload(client)

    with tab_list:
        _render_list(client)


# ── Upload ────────────────────────────────────────────────────────────────────

def _render_upload(client: APIClient) -> None:
    st.subheader("Upload Document")
    st.caption("Supported: PDF, DOCX, TXT, Markdown, HTML")

    files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    with st.expander("📝 Metadata (optional)"):
        col1, col2 = st.columns(2)
        with col1:
            title        = st.text_input("Title", placeholder="Auto-detected if blank")
            department   = st.text_input("Department", placeholder="e.g. HR, Engineering")
        with col2:
            doc_category = st.text_input("Category", placeholder="e.g. policy, SOP, manual")
            tags         = st.text_input("Tags", placeholder="comma-separated")

    if st.button("🚀 Upload & Ingest", type="primary", disabled=not files):
        progress = st.progress(0, text="Uploading…")
        results  = []

        for i, f in enumerate(files):
            progress.progress((i + 1) / len(files), text=f"Uploading {f.name}…")
            data, err = client.upload_document(
                file_bytes=f.read(),
                filename=f.name,
                title=title,
                department=department,
                doc_category=doc_category,
                tags=tags,
            )
            results.append((f.name, data, err))

        progress.empty()

        for fname, data, err in results:
            if err:
                st.error(f"❌ **{fname}**: {err}")
            else:
                doc_id = data.get("document_id", "")
                job_id = data.get("job_id", "")
                st.success(
                    f"✅ **{fname}** accepted  \n"
                    f"`doc_id: {doc_id[:8]}…`  `job_id: {job_id[:8]}…`"
                )

        if any(not err for _, _, err in results):
            st.info("ℹ️ Ingestion runs in the background. Check the **All Documents** tab for status.")


# ── Document list ─────────────────────────────────────────────────────────────

def _render_list(client: APIClient) -> None:
    st.subheader("Documents")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("Search by name/title", placeholder="🔍 filename or title")
    with col2:
        status_filter = st.selectbox(
            "Status", ["all", "pending", "processing", "indexed", "failed"]
        )
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    data, err = client.list_documents(
        page=1, page_size=50,
        status="" if status_filter == "all" else status_filter,
        search=search,
    )

    if err:
        st.error(f"❌ {err}")
        return

    docs  = data if isinstance(data, list) else data.get("data", [])
    meta  = data.get("meta", {}) if isinstance(data, dict) else {}
    total = meta.get("total", len(docs))

    st.caption(f"{total} document(s)")

    if not docs:
        st.info("No documents found. Upload some files to get started.")
        return

    for doc in docs:
        _render_doc_row(client, doc)


def _render_doc_row(client: APIClient, doc: dict) -> None:
    status = doc.get("status", "unknown")
    icon   = _STATUS_COLOUR.get(status, "⚪")
    doc_id = doc.get("id", "")

    with st.container(border=True):
        col1, col2, col3 = st.columns([5, 2, 1])

        with col1:
            st.markdown(f"{icon} **{doc.get('filename', 'Unknown')}**")
            meta_parts = []
            if doc.get("title") and doc["title"] != doc.get("filename"):
                meta_parts.append(doc["title"])
            if doc.get("department"):
                meta_parts.append(f"🏢 {doc['department']}")
            if doc.get("chunk_count"):
                meta_parts.append(f"📄 {doc['chunk_count']} chunks")
            if meta_parts:
                st.caption("  ·  ".join(meta_parts))

        with col2:
            st.caption(f"Status: **{status}**")
            created = doc.get("created_at", "")[:10]
            if created:
                st.caption(f"📅 {created}")

        with col3:
            if st.button("🗑", key=f"del_{doc_id}", help="Delete document"):
                ok, err = client.delete_document(doc_id)
                if ok:
                    st.toast(f"Deleted {doc.get('filename')}", icon="🗑")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(err)

        # Status polling for in-progress docs
        if status in ("pending", "processing"):
            with st.spinner(f"Ingesting… (status: {status})"):
                status_data, _ = client.get_document_status(doc_id)
                if status_data:
                    progress = status_data.get("progress_pct") or 0
                    if progress:
                        st.progress(progress / 100, text=f"{progress:.0f}%")
