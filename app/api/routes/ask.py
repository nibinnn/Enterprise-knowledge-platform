"""app/api/routes/ask.py — RAG question-answering endpoints."""
from __future__ import annotations
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_agent_service, get_ask_service,
    get_current_user, get_db,
)
from app.api.schemas.ask import AgentAskResponse, AskRequest, AskResponse
from app.api.schemas.auth import CurrentUser
from app.api.schemas.common import APIResponse
from app.services.agent import AgentService
from app.services.ask import AskService

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post(
    "/",
    response_model=APIResponse[AskResponse],
    summary="Ask a question — standard RAG",
)
async def ask(
    body:         AskRequest,
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    svc:          AskService   = Depends(get_ask_service),
):
    """
    Ask a natural-language question. Returns a grounded answer with citations.

    Pipeline: query → retrieval → rerank → context → LLM → answer + citations

    Set **mode='agent'** to activate multi-step reasoning (slower but handles
    complex, multi-part questions).

    Set **stream=true** to receive a Server-Sent Events stream.
    """
    if body.stream:
        return await _stream_response(body, svc, db)

    if body.mode == "agent":
        raise HTTPException(
            status_code=400,
            detail="Use POST /ask/agent for agentic mode, or set mode='rag'.",
        )

    result = await svc.ask(body, db)
    return APIResponse(data=result)


@router.post(
    "/agent",
    response_model=APIResponse[AgentAskResponse],
    summary="Ask a question — agentic research mode",
)
async def ask_agent(
    body:         AskRequest,
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    svc:          AgentService = Depends(get_agent_service),
):
    """
    Multi-step agent that decomposes complex questions, calls multiple tools,
    and synthesises a final answer with a full reasoning trace.

    Returns **steps[]** alongside the answer so you can inspect the
    agent's reasoning path.
    """
    result = await svc.run(body, db)
    return APIResponse(data=result)


@router.get(
    "/history",
    summary="Retrieve recent answers (placeholder)",
)
async def ask_history(
    limit:        int = 20,
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Return recent answers for the current session.
    Stub — requires Answer persistence table (Day 16).
    """
    return APIResponse(data=[])


# ── Streaming helper ──────────────────────────────────────────────────────────

async def _stream_response(
    body: AskRequest, svc: AskService, db
) -> StreamingResponse:
    """
    Server-Sent Events streaming for real-time answer generation.
    Each event is a JSON-encoded delta or the final AskResponse.
    """
    async def generator() -> AsyncGenerator[str, None]:
        # TODO (Day 16): wire up LLM streaming
        # For now, yield the full answer in one event
        result = await svc.ask(body, db)
        payload = json.dumps(result.model_dump(), default=str)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
