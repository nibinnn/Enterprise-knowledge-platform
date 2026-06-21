"""
app/services/ask.py
Stub until Day 16 (full RAG pipeline: retriever → reranker → context → LLM).
"""
from __future__ import annotations
import time, uuid
from datetime import datetime
from typing import List
from app.api.schemas.ask import AskRequest, AskResponse, AgentAskResponse, CitationOut
from app.config import get_settings

settings = get_settings()


class AskService:

    async def ask(self, request: AskRequest, db) -> AskResponse:
        """
        Answer a question using RAG.
        Stub returns a placeholder answer.
        Day 16 replaces with:
            retriever.retrieve() → reranker → context_builder → llm.generate()
        """
        start = time.perf_counter()

        # TODO (Day 16): full RAG pipeline
        answer_text = (
            "The knowledge base is being indexed. "
            "Please try again once documents have been ingested."
        )

        return AskResponse(
            answer_id=str(uuid.uuid4()),
            question=request.question,
            answer=answer_text,
            citations=[],
            model_used=settings.llm_model,
            total_latency_ms=round((time.perf_counter() - start) * 1000, 2),
            created_at=datetime.utcnow(),
        )

    async def ask_agent(self, request: AskRequest, db) -> AgentAskResponse:
        """
        Answer using the multi-step agent (research mode).
        Stub until Block 2 (agent layer).
        """
        base = await self.ask(request, db)
        return AgentAskResponse(**base.model_dump(), steps=[], total_steps=0)
