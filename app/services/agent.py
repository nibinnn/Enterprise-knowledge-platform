"""
app/services/agent.py
Agent service wired to the full AgentOrchestrator (tool-calling loop).
"""
from __future__ import annotations

import json

from app.api.schemas.ask import AgentAskResponse, AgentStepOut, AskRequest
from app.config import get_settings

settings = get_settings()


class AgentService:

    async def run(self, request: AskRequest, db) -> AgentAskResponse:
        from app.agents.orchestrator import AgentOrchestrator

        filters: dict | None = None
        if request.filters:
            d = {}
            if request.filters.department:
                d["department"] = request.filters.department
            if request.filters.doc_category:
                d["doc_category"] = request.filters.doc_category
            if request.filters.file_type:
                d["doc_type"] = request.filters.file_type
            if request.filters.tags:
                d["tags"] = request.filters.tags
            filters = d or None

        orchestrator = AgentOrchestrator()
        agent_resp = await orchestrator.run(
            question=request.question,
            filters=filters,
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
