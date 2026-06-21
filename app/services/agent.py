"""
app/services/agent.py
Stub until Block 2 (Days 11-13: agent tools + orchestrator).
"""
from __future__ import annotations
from app.api.schemas.ask import AskRequest, AgentAskResponse
from app.services.ask import AskService


class AgentService:

    async def run(self, request: AskRequest, db) -> AgentAskResponse:
        """
        Run the multi-step agent research loop.
        Delegates to AskService stub until Block 2.
        """
        ask_svc = AskService()
        return await ask_svc.ask_agent(request, db)
