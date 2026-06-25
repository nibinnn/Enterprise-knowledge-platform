"""app/agents/tools/analytics_tool.py — cross-document analytics."""
from __future__ import annotations
from app.core.base.tool import BaseTool, ToolResult


class AnalyticsTool(BaseTool):
    """Count, aggregate, or find patterns across many documents."""

    name        = "analyze_knowledge_base"
    description = (
        "Run analytics across the knowledge base: count documents matching criteria, "
        "find which departments have the most content, identify common topics, "
        "or get corpus statistics. Use for questions like 'how many policies mention X?' "
        "or 'which departments have SOPs about Y?'"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query":        {"type": "string", "description": "Topic or keyword to analyze."},
            "analysis_type": {
                "type": "string",
                "enum": ["count", "by_department", "by_doc_type", "corpus_stats"],
                "default": "count",
            },
            "department": {"type": "string", "description": "Filter by department (optional)."},
        },
        "required": ["query"],
    }

    def __init__(self, retriever=None, db=None):
        self._retriever = retriever
        self._db        = db

    def _run(self, query: str, analysis_type: str = "count", department: str = None) -> ToolResult:
        import asyncio
        try:
            return asyncio.run(self._arun(query=query, analysis_type=analysis_type, department=department))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(
                self._arun(query=query, analysis_type=analysis_type, department=department)
            )

    async def _arun(self, query: str, analysis_type: str = "count", department: str = None) -> ToolResult:
        if analysis_type == "corpus_stats":
            return await self._corpus_stats()

        # Use the retriever to find matching chunks
        retriever = self._get_retriever()
        filters   = {"department": department} if department else None
        chunks    = await retriever.retrieve(query=query, top_k=50, filters=filters)

        if analysis_type == "count":
            doc_ids = set(c.metadata.doc_id for c in chunks)
            data    = {
                "query": query,
                "matching_documents": len(doc_ids),
                "matching_chunks":    len(chunks),
                "department_filter":  department,
            }
            output = f"Found {len(doc_ids)} document(s) and {len(chunks)} passage(s) related to '{query}'."
            return ToolResult.ok(self.name, output, data=data)

        if analysis_type == "by_department":
            from collections import Counter
            dept_counts = Counter(c.metadata.department or "Unknown" for c in chunks)
            data = {"query": query, "by_department": dict(dept_counts.most_common())}
            output = f"Topic '{query}' distribution: " + ", ".join(f"{d}={n}" for d, n in dept_counts.most_common(5))
            return ToolResult.ok(self.name, output, data=data)

        if analysis_type == "by_doc_type":
            from collections import Counter
            type_counts = Counter(c.metadata.doc_type.value for c in chunks)
            data   = {"query": query, "by_doc_type": dict(type_counts.most_common())}
            output = f"Topic '{query}' by doc type: " + ", ".join(f"{t}={n}" for t, n in type_counts.most_common())
            return ToolResult.ok(self.name, output, data=data)

        return ToolResult.fail(self.name, f"Unknown analysis_type: {analysis_type}")

    async def _corpus_stats(self) -> ToolResult:
        if self._db is None:
            return ToolResult.ok(self.name, "Corpus stats require a DB session.", data={})
        from sqlalchemy import func, select
        from app.db.models import DocumentModel, ChunkModel
        total_docs   = (await self._db.execute(select(func.count(DocumentModel.id)))).scalar_one()
        indexed_docs = (await self._db.execute(
            select(func.count(DocumentModel.id)).where(DocumentModel.status == "indexed")
        )).scalar_one()
        total_chunks = (await self._db.execute(select(func.count(ChunkModel.id)))).scalar_one()
        data = {"total_documents": total_docs, "indexed_documents": indexed_docs, "total_chunks": total_chunks}
        return ToolResult.ok(self.name, f"Corpus: {indexed_docs}/{total_docs} docs indexed, {total_chunks} chunks.", data=data)

    def _get_retriever(self):
        if self._retriever is None:
            from app.retrieval.retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever
