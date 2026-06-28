"""
frontend/components/admin.py
Admin dashboard: system stats, service health, evaluation runs.
"""
from __future__ import annotations

import streamlit as st

from frontend.utils.api_client import APIClient


def render_admin(client: APIClient) -> None:
    st.subheader("Admin Dashboard")
    tab_stats, tab_health, tab_eval = st.tabs(["📊 Stats", "🏥 Health", "🧪 Evaluation"])

    with tab_stats:
        _render_stats(client)
    with tab_health:
        _render_health(client)
    with tab_eval:
        _render_eval(client)


def _render_stats(client: APIClient) -> None:
    if st.button("🔄 Refresh", key="refresh_stats"):
        st.rerun()

    data, err = client.get_stats()
    if err:
        st.error(f"❌ {err}")
        return

    # Document counts
    docs = data.get("documents", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", docs.get("total", 0))
    by_status = docs.get("by_status", {})
    col2.metric("✅ Indexed",   by_status.get("indexed",    0))
    col3.metric("⏳ Pending",   by_status.get("pending",    0))
    col4.metric("❌ Failed",    by_status.get("failed",     0))

    st.divider()
    col5, col6 = st.columns(2)
    col5.metric("Total Chunks", data.get("chunks", {}).get("total", 0))
    col6.metric("Feedback Entries", data.get("feedback", {}).get("total", 0))

    # Config
    cfg = data.get("config", {})
    if cfg:
        st.subheader("Current Configuration")
        rows = {
            "Embedding Provider": cfg.get("embedding_provider", "—"),
            "Embedding Model":    cfg.get("embedding_model",    "—"),
            "LLM Provider":       cfg.get("llm_provider",       "—"),
            "LLM Model":          cfg.get("llm_model",          "—"),
            "Chunking Strategy":  cfg.get("chunking_strategy",  "—"),
            "Search Mode":        cfg.get("search_mode",        "—"),
        }
        for k, v in rows.items():
            col_k, col_v = st.columns([2, 3])
            col_k.caption(k)
            col_v.markdown(f"`{v}`")


def _render_health(client: APIClient) -> None:
    if st.button("🔄 Check Health", key="refresh_health"):
        st.rerun()

    data, err = client.get_health()
    if err:
        st.error(f"❌ {err}")
        return

    overall = data.get("status", "unknown")
    if overall == "healthy":
        st.success("🟢 All systems healthy")
    else:
        st.warning("🟡 Some services degraded")

    checks = data.get("checks", {})
    for svc, status in checks.items():
        icon = "🟢" if status == "ok" else "🔴"
        col1, col2 = st.columns([2, 5])
        col1.markdown(f"{icon} **{svc.title()}**")
        col2.caption(status)


def _render_eval(client: APIClient) -> None:
    st.markdown("#### Start Evaluation Run")
    col1, col2 = st.columns(2)
    with col1:
        run_name = st.text_input("Run name", placeholder="e.g. baseline-v1")
    with col2:
        dataset  = st.text_input("Dataset name", placeholder="e.g. hr_qa_50")

    if st.button("▶️ Start Eval Run", type="primary", disabled=not (run_name and dataset)):
        data, err = client.start_eval_run(run_name, dataset)
        if err:
            st.error(f"❌ {err}")
        else:
            st.success(f"✅ Eval run started: `{data.get('id', '')[:8]}…`")
            st.rerun()

    st.divider()
    st.markdown("#### Previous Runs")

    data, err = client.list_eval_runs()
    if err:
        st.error(f"❌ {err}")
        return

    runs = data if isinstance(data, list) else data.get("data", [])
    if not runs:
        st.info("No evaluation runs yet.")
        return

    for run in runs:
        with st.container(border=True):
            st.markdown(f"**{run.get('run_name')}**  ·  `{run.get('dataset_name')}`")
            metrics = run.get("metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            def _m(val): return f"{val:.3f}" if val is not None else "—"
            c1.metric("Faithfulness",      _m(metrics.get("faithfulness")))
            c2.metric("Answer Relevance",  _m(metrics.get("answer_relevance")))
            c3.metric("Context Precision", _m(metrics.get("context_precision")))
            c4.metric("Context Recall",    _m(metrics.get("context_recall")))
            st.caption(
                f"Questions: {run.get('num_questions', 0)}  ·  "
                f"Model: {run.get('llm_model', '—')}  ·  "
                f"Created: {str(run.get('created_at', ''))[:10]}"
            )
