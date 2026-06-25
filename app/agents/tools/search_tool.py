"""app/agents/tools/search_tool.py — SearchTool for the agent layer."""
from __future__ import annotations
import json
from typing import Any, Dict, List
from app.core.base.tool import BaseTool, ToolResult


class SearchTool(BaseTool):
    """Search the knowledge base and return the most relevant text chunks."""

    name        = "search_knowledge_base"
    description = (
        "Search the enterprise knowledge base using semantic + keyword hybrid search. "
        "Use this to find relevant passages, policies, procedures, or facts. "
        "Returns the top matching text chunks with source metadata."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {"type": "integer", "description": "Number of results (default 5, max 20).", "default": 5},
            "department": {"type": "string", "description": "Filter by department (optional)."},
            "doc_category": {"type": "string", "description": "Filter by doc category (optional)."},
        },
        "required": ["query"],
    }

    def __init__(self, retriever=None):
        self._retriever = retriever

    def _run(self, query: str, top_k: int = 5, department: str = None, doc_category: str = None) -> ToolResult:
        import asyncio
        try:
            result = asyncio.run(self._arun_impl(query, top_k, department, doc_category))
            return result
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun_impl(query, top_k, department, doc_category))

    async def _arun(self, query: str, top_k: int = 5, department: str = None, doc_category: str = None) -> ToolResult:
        return await self._arun_impl(query, top_k, department, doc_category)

    async def _arun_impl(self, query: str, top_k: int = 5, department: str = None, doc_category: str = None) -> ToolResult:
        retriever = self._get_retriever()
        filters: Dict[str, Any] = {}
        if department:
            filters["department"] = department
        if doc_category:
            filters["doc_category"] = doc_category

        chunks = await retriever.retrieve(query=query, top_k=min(top_k, 20), filters=filters or None)
        if not chunks:
            return ToolResult.ok(self.name, "No relevant information found.", data={"results": []})

        results = []
        for c in chunks:
            results.append({
                "chunk_id": c.chunk_id,
                "text": c.text[:800],
                "score": round(c.score, 4),
                "source": c.metadata.doc_filename,
                "page": c.metadata.page_number,
                "section": c.metadata.section_heading,
            })

        summary = f"Found {len(results)} relevant passage(s) for '{query}'."
        return ToolResult.ok(self.name, summary, data={"results": results, "query": query})

    def _get_retriever(self):
        if self._retriever is None:
            from app.retrieval.retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever
