"""app/api/routes — all route modules registered in main.py."""
from app.api.routes import auth, documents, search, ask, feedback, admin

__all__ = ["auth", "documents", "search", "ask", "feedback", "admin"]
