"""
app/services/ask.py
RAG ask service wired to the full pipeline: retriever → reranker → context → LLM → answer.
"""
from __future__ import annotations

import json
from typing import List, Optional

from app.api.schemas.ask import AgentAskResponse, AgentStepOut, AskRequest, AskResponse, CitationOut
from app.config import get_settings

settings = get_settings()


def _filters_to_dict(filters) -> Optional[dict]:
    if filters is None:
        return None
    d: dict = {}
    if filters.department:
        d["department"] = filters.department
    if filters.doc_category:
        d["doc_category"] = filters.doc_category
    if filters.file_type:
        d["doc_type"] = filters.file_type
    if filters.tags:
        d["tags"] = filters.tags
    return d or None


def _map_citations(citations) -> List[CitationOut]:
    return [
        CitationOut(
            citation_id=c.citation_id,
            chunk_id=c.chunk_id,
            document_id=c.doc_id,
            filename=c.doc_filename,
            page_number=c.page_number,
            section_heading=c.section_heading,
            excerpt=c.excerpt,
        )
        for c in citations
    ]


class AskService:

    async def ask(self, request: AskRequest, db) -> AskResponse:
        from app.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
        )
        answer = await pipeline.run(
            question=request.question,
            filters=_filters_to_dict(request.filters),
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
        )
        return AskResponse(
            answer_id=answer.id,
            question=answer.query,
            answer=answer.answer_text,
            citations=_map_citations(answer.citations),
            confidence=answer.confidence,
            model_used=answer.model_used,
            retrieval_latency_ms=answer.retrieval_latency_ms,
            llm_latency_ms=answer.llm_latency_ms,
            total_latency_ms=answer.total_latency_ms,
            created_at=answer.created_at,
        )

    async def ask_agent(self, request: AskRequest, db) -> AgentAskResponse:
        from app.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()
        agent_resp = await orchestrator.run(
            question=request.question,
            filters=_filters_to_dict(request.filters),
        )
        steps = [
            AgentStepOut(
                step=s.step,
                tool_name=s.tool_name,
                tool_input=json.dumps(s.tool_input, default=str),
                tool_output=s.tool_output,
                latency_ms=s.latency_ms,
            )
            for s in agent_resp.steps
        ]
        return AgentAskResponse(
            answer_id=agent_resp.answer_id,
            question=request.question,
            answer=agent_resp.answer,
            citations=[],
            model_used=settings.llm_model,
            total_latency_ms=agent_resp.total_ms,
            created_at=agent_resp.created_at,
            steps=steps,
            total_steps=agent_resp.total_steps,
        )
