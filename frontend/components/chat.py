"""
frontend/components/chat.py
Chat interface: ask questions, stream answers, display citations, collect feedback.
"""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from frontend.utils.api_client import APIClient


def _init_history() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _badge(text: str, color: str = "#e8f4fd") -> str:
    return (
        f'<span style="background:{color};padding:2px 8px;border-radius:12px;'
        f'font-size:0.75rem;font-weight:600;">{text}</span>'
    )


def render_chat(client: APIClient) -> None:
    _init_history()

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        mode = st.radio(
            "Mode", ["RAG (fast)", "Agent (deep)"],
            horizontal=True, label_visibility="collapsed",
            help="RAG: single retrieval pass. Agent: multi-step reasoning with tools.",
        )
    with col2:
        top_k = st.number_input("Retrieve", min_value=1, max_value=50, value=20, step=5,
                                help="Chunks fetched before reranking")
    with col3:
        rerank_n = st.number_input("Rerank to", min_value=1, max_value=20, value=5, step=1,
                                   help="Chunks kept after reranking")

    with st.expander("🔽 Filters (optional)"):
        f_dept = st.text_input("Department filter", placeholder="e.g. HR, Engineering")
        f_cat  = st.text_input("Category filter",   placeholder="e.g. policy, SOP")

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            _render_message(msg)

    # ── Input ─────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents…"):
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_history.append(user_msg)

        with chat_container:
            _render_message(user_msg)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching knowledge base…"):
                    t0   = time.perf_counter()
                    data, err = client.ask(
                        question=prompt,
                        mode="agent" if "Agent" in mode else "rag",
                        top_k=top_k,
                        rerank_top_n=rerank_n,
                        department=f_dept,
                        doc_category=f_cat,
                    )
                    elapsed = time.perf_counter() - t0

                if err:
                    st.error(f"❌ {err}")
                    return

                # Answer
                st.markdown(data.get("answer", ""))

                # Latency badge
                total_ms = data.get("total_latency_ms") or elapsed * 1000
                st.markdown(
                    _badge(f"⏱ {total_ms:.0f} ms") + "  " +
                    _badge(data.get("model_used", ""), "#f0f0f0"),
                    unsafe_allow_html=True,
                )

                # Agent steps (if agent mode)
                steps = data.get("steps", [])
                if steps:
                    with st.expander(f"🔍 Reasoning trace ({len(steps)} steps)"):
                        for s in steps:
                            st.markdown(
                                f"**Step {s['step']} — `{s['tool_name']}`** "
                                f"_{s.get('latency_ms', 0):.0f} ms_"
                            )
                            st.caption(f"Input: {s.get('tool_input', '')}")
                            st.text(s.get("tool_output", "")[:400])

                # Citations
                citations = data.get("citations", [])
                if citations:
                    with st.expander(f"📚 Sources ({len(citations)})"):
                        for i, c in enumerate(citations, 1):
                            parts = [f"**[{i}] {c['filename']}**"]
                            if c.get("page_number"):
                                parts.append(f"p. {c['page_number']}")
                            if c.get("section_heading"):
                                parts.append(f"§ {c['section_heading']}")
                            st.markdown("  ·  ".join(parts))
                            if c.get("excerpt"):
                                st.caption(f"> {c['excerpt'][:300]}…")
                            st.divider()

                # Feedback row
                answer_id = data.get("answer_id", "")
                if answer_id:
                    _render_feedback(client, answer_id)

        # Save to history
        assistant_msg = {
            "role": "assistant",
            "content": data.get("answer", ""),
            "data": data,
        }
        st.session_state.chat_history.append(assistant_msg)

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


def _render_message(msg: dict) -> None:
    role    = msg["role"]
    content = msg["content"]
    avatar  = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        # Re-render citations for history messages
        data = msg.get("data", {})
        citations = data.get("citations", [])
        if citations:
            with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
                for i, c in enumerate(citations, 1):
                    st.caption(f"[{i}] {c['filename']}" +
                               (f" p.{c['page_number']}" if c.get("page_number") else ""))


def _render_feedback(client: APIClient, answer_id: str) -> None:
    key_base = f"fb_{answer_id}"
    if st.session_state.get(f"{key_base}_submitted"):
        st.success("✅ Feedback recorded — thank you!")
        return

    cols = st.columns([1, 1, 1, 1, 1, 4])
    labels = ["😤 1", "😕 2", "😐 3", "🙂 4", "😄 5"]
    for i, (col, label) in enumerate(zip(cols[:5], labels)):
        with col:
            if st.button(label, key=f"{key_base}_r{i+1}", use_container_width=True):
                _, err = client.submit_feedback(answer_id, rating=i + 1)
                if not err:
                    st.session_state[f"{key_base}_submitted"] = True
                    st.rerun()
