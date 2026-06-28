"""
frontend/components/search.py
Direct semantic/keyword/hybrid search panel.
"""
from __future__ import annotations

import streamlit as st

from frontend.utils.api_client import APIClient

_MODE_HELP = {
    "hybrid":  "Combines semantic + keyword search using Reciprocal Rank Fusion (recommended)",
    "dense":   "Pure semantic vector search — best for conceptual queries",
    "keyword": "BM25 full-text search — best for exact term matching",
}

_SCORE_COLOR = {
    (0.8, 1.1): "#d4edda",
    (0.6, 0.8): "#fff3cd",
    (0.0, 0.6): "#f8d7da",
}


def _score_bg(score: float) -> str:
    for (lo, hi), color in _SCORE_COLOR.items():
        if lo <= score < hi:
            return color
    return "#f8f9fa"


def render_search(client: APIClient) -> None:
    st.subheader("Knowledge Base Search")

    # ── Query bar ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Search query", placeholder="Type to search…", label_visibility="collapsed")
    with col2:
        search_btn = st.button("Search 🔍", type="primary", use_container_width=True)

    # ── Options ───────────────────────────────────────────────────────────────
    with st.expander("⚙️ Options"):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            mode  = st.selectbox("Mode", ["hybrid", "dense", "keyword"],
                                 help=_MODE_HELP.get("hybrid"))
        with col_b:
            top_k = st.slider("Results", 1, 30, 10)
        with col_c:
            dept  = st.text_input("Department", placeholder="filter…")
        with col_d:
            cat   = st.text_input("Category", placeholder="filter…")

    # ── Execute ───────────────────────────────────────────────────────────────
    if search_btn and query:
        with st.spinner(f"Searching ({mode})…"):
            data, err = client.search(
                query=query, mode=mode, top_k=top_k,
                department=dept, doc_category=cat,
            )

        if err:
            st.error(f"❌ {err}")
            return

        results    = data.get("results", []) if isinstance(data, dict) else []
        latency_ms = data.get("latency_ms", 0) if isinstance(data, dict) else 0
        total      = data.get("total_results", len(results)) if isinstance(data, dict) else len(results)

        st.caption(f"Found **{total}** result(s) in {latency_ms:.0f} ms  ·  mode: `{mode}`")

        if not results:
            st.info("No results. Try broadening your query or switching to **hybrid** mode.")
            return

        for i, r in enumerate(results, 1):
            _render_result(i, r)


def _render_result(rank: int, r: dict) -> None:
    score = r.get("score", 0)
    bg    = _score_bg(score)
    filename = r.get("filename", "Unknown")
    page     = r.get("page_number")
    section  = r.get("section_heading")
    text     = r.get("text", "")

    with st.container(border=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            location_parts = [f"**#{rank}  {filename}**"]
            if page:    location_parts.append(f"p. {page}")
            if section: location_parts.append(f"§ {section}")
            st.markdown("  ·  ".join(location_parts))
            if r.get("department"):
                st.caption(f"🏢 {r['department']}")
        with col2:
            st.markdown(
                f'<div style="background:{bg};border-radius:8px;'
                f'text-align:center;padding:6px 0;font-weight:700;">'
                f'{score:.3f}</div>',
                unsafe_allow_html=True,
            )

        # Text preview — highlight query terms
        preview = text[:600] + ("…" if len(text) > 600 else "")
        st.text(preview)
