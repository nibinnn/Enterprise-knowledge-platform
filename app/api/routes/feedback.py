"""app/api/routes/feedback.py — user feedback and evaluation endpoints."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.schemas.auth import CurrentUser
from app.api.schemas.common import APIResponse, PaginatedResponse, PaginationMeta
from app.api.schemas.eval import EvalRunOut, EvalRunRequest, EvalMetricsOut
from app.api.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.db.models import EvalRunModel, FeedbackModel

router = APIRouter(tags=["feedback & eval"])


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=APIResponse[FeedbackResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback on an answer",
)
async def submit_feedback(
    body:         FeedbackRequest,
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Submit a rating (1-5) and optional correction for an answer.
    Feedback is stored and used as signal for re-ranking improvements.
    """
    row = FeedbackModel(
        id=str(uuid.uuid4()),
        answer_id=body.answer_id,
        query="",                         # populated from answer_id lookup on Day 16
        answer_text="",
        rating=body.rating,
        correction=body.correction,
        bad_citation_ids=body.bad_citation_ids,
        extra_json={"comment": body.comment} if body.comment else None,
    )
    db.add(row)
    await db.flush()

    return APIResponse(
        data=FeedbackResponse(
            feedback_id=row.id,
            answer_id=row.answer_id,
            rating=row.rating,
            created_at=row.created_at or datetime.utcnow(),
        )
    )


@router.get(
    "/feedback",
    response_model=PaginatedResponse[FeedbackResponse],
    summary="List feedback entries",
)
async def list_feedback(
    page:         int = Query(default=1, ge=1),
    page_size:    int = Query(default=20, ge=1, le=100),
    answer_id:    str = Query(default=None),
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    from sqlalchemy import func
    q = select(FeedbackModel)
    if answer_id:
        q = q.where(FeedbackModel.answer_id == answer_id)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows  = (await db.execute(
        q.order_by(FeedbackModel.created_at.desc())
         .offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()

    return PaginatedResponse(
        data=[
            FeedbackResponse(
                feedback_id=r.id,
                answer_id=r.answer_id,
                rating=r.rating,
                created_at=r.created_at or datetime.utcnow(),
            )
            for r in rows
        ],
        meta=PaginationMeta(
            total=total, page=page, page_size=page_size,
            total_pages=max(1, -(-total // page_size)),
        ),
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

@router.post(
    "/eval/runs",
    response_model=APIResponse[EvalRunOut],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an evaluation run",
)
async def start_eval_run(
    body:         EvalRunRequest,
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Kick off a RAGAS evaluation run against a ground-truth dataset.
    Returns immediately — poll GET /eval/runs/{id} for results.
    Real implementation wired on Day 18.
    """
    from app.config import get_settings
    s = get_settings()
    row = EvalRunModel(
        id=str(uuid.uuid4()),
        run_name=body.run_name,
        dataset_name=body.dataset_name,
        llm_model=s.llm_model,
        embedding_model=s.embedding_model,
        chunking_strategy=s.chunking_strategy.value,
        num_questions=len(body.questions) if body.questions else 0,
    )
    db.add(row)
    await db.flush()

    # TODO (Day 18): dispatch eval Celery task
    return APIResponse(data=_eval_row_to_out(row))


@router.get(
    "/eval/runs",
    response_model=PaginatedResponse[EvalRunOut],
    summary="List evaluation runs",
)
async def list_eval_runs(
    page:         int = Query(default=1, ge=1),
    page_size:    int = Query(default=10, ge=1, le=50),
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    from sqlalchemy import func
    q     = select(EvalRunModel)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows  = (await db.execute(
        q.order_by(EvalRunModel.created_at.desc())
         .offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()

    return PaginatedResponse(
        data=[_eval_row_to_out(r) for r in rows],
        meta=PaginationMeta(
            total=total, page=page, page_size=page_size,
            total_pages=max(1, -(-total // page_size)),
        ),
    )


@router.get(
    "/eval/runs/{run_id}",
    response_model=APIResponse[EvalRunOut],
    summary="Get evaluation run result",
)
async def get_eval_run(
    run_id:       str,
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    result = await db.execute(select(EvalRunModel).where(EvalRunModel.id == run_id))
    row    = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found.")
    return APIResponse(data=_eval_row_to_out(row))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eval_row_to_out(row: EvalRunModel) -> EvalRunOut:
    return EvalRunOut(
        id=row.id,
        run_name=row.run_name,
        dataset_name=row.dataset_name,
        metrics=EvalMetricsOut(
            faithfulness=row.faithfulness,
            answer_relevance=row.answer_relevance,
            context_precision=row.context_precision,
            context_recall=row.context_recall,
        ),
        llm_model=row.llm_model,
        embedding_model=row.embedding_model,
        chunking_strategy=row.chunking_strategy,
        num_questions=row.num_questions,
        created_at=row.created_at or datetime.utcnow(),
    )
