"""
frontend/app.py
─────────────────────────────────────────────────────────────────────────────
Enterprise Knowledge Intelligence Platform — Streamlit Frontend

Run with:
    streamlit run frontend/app.py

Environment:
    API_BASE_URL   URL of the FastAPI backend (default: http://localhost:8000/api/v1)
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EKIP — Knowledge Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.utils.api_client import APIClient
from frontend.components.chat             import render_chat
from frontend.components.document_manager import render_document_manager
from frontend.components.search           import render_search
from frontend.components.admin            import render_admin


# ── Session-state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    if "client"        not in st.session_state:
        st.session_state.client = APIClient()
    if "active_page"   not in st.session_state:
        st.session_state.active_page = "💬 Chat"
    if "chat_history"  not in st.session_state:
        st.session_state.chat_history = []


_init_state()
client: APIClient = st.session_state.client


# ── Login wall ────────────────────────────────────────────────────────────────
def _render_login() -> None:
    st.title("🧠 EKIP")
    st.caption("Enterprise Knowledge Intelligence Platform")
    st.divider()

    with st.form("login_form"):
        col1, col2 = st.columns([3, 2])
        with col1:
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            api_url  = st.text_input("API URL", value="http://localhost:8000/api/v1")
        submitted = st.form_submit_button("Sign In →", type="primary", use_container_width=False)

    if submitted:
        client._base_url = api_url  # allow override
        ok, msg = client.login(username, password)
        if ok:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error(f"❌ {msg}")


if not client.is_authenticated():
    _render_login()
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 EKIP")
    st.caption("Enterprise Knowledge Intelligence")
    st.divider()

    pages = ["💬 Chat", "📁 Documents", "🔍 Search", "⚙️ Admin"]
    for page in pages:
        if st.button(page, use_container_width=True,
                     type="primary" if st.session_state.active_page == page else "secondary"):
            st.session_state.active_page = page
            st.rerun()

    st.divider()

    # Quick stats in sidebar
    stats, _ = client.get_stats()
    if stats:
        docs = stats.get("documents", {})
        st.metric("Indexed Documents", docs.get("by_status", {}).get("indexed", 0))
        st.metric("Total Chunks",      stats.get("chunks", {}).get("total", 0))

    st.divider()
    if st.button("Sign Out", use_container_width=True):
        client.logout()
        st.session_state.authenticated = False
        st.session_state.chat_history  = []
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
page = st.session_state.active_page

if page == "💬 Chat":
    st.title("💬 Ask Your Knowledge Base")
    render_chat(client)

elif page == "📁 Documents":
    st.title("📁 Document Management")
    render_document_manager(client)

elif page == "🔍 Search":
    st.title("🔍 Search")
    render_search(client)

elif page == "⚙️ Admin":
    st.title("⚙️ Admin")
    render_admin(client)
