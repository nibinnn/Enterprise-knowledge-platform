"""app/api/routes/admin.py — system stats, health details, and config endpoints."""
from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import require_scope
from app.api.dependencies import get_current_user, get_db
from app.api.schemas.auth import CurrentUser
from app.api.schemas.common import APIResponse
from app.config import get_settings
from app.db.database import check_db_connection
from app.db.models import ChunkModel, DocumentModel, FeedbackModel, JobModel

router   = APIRouter(prefix="/admin", tags=["admin"])
settings = get_settings()


@router.get("/stats", response_model=APIResponse[Dict[str, Any]], summary="System statistics")
async def system_stats(
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Return aggregate statistics: document counts, chunk counts, job statuses."""
    require_scope(current_user, "admin")

    doc_counts = dict(
        (await db.execute(
            select(DocumentModel.status, func.count())
            .group_by(DocumentModel.status)
        )).all()
    )

    total_chunks   = (await db.execute(select(func.count(ChunkModel.id)))).scalar_one()
    total_feedback = (await db.execute(select(func.count(FeedbackModel.id)))).scalar_one()

    job_counts = dict(
        (await db.execute(
            select(JobModel.status, func.count())
            .group_by(JobModel.status)
        )).all()
    )

    return APIResponse(data={
        "documents": {
            "total":      sum(doc_counts.values()),
            "by_status":  doc_counts,
        },
        "chunks":        {"total": total_chunks},
        "jobs":          {"by_status": job_counts},
        "feedback":      {"total": total_feedback},
        "config": {
            "embedding_provider": settings.embedding_provider.value,
            "embedding_model":    settings.embedding_model,
            "llm_provider":       settings.llm_provider.value,
            "llm_model":          settings.llm_model,
            "chunking_strategy":  settings.chunking_strategy.value,
            "search_mode":        settings.search_mode.value,
        },
    })


@router.get("/health/detailed", response_model=APIResponse[Dict[str, Any]], summary="Detailed health check")
async def detailed_health(
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Check all downstream service connections."""
    require_scope(current_user, "admin")

    checks: Dict[str, Any] = {}

    # Postgres
    checks["postgres"] = "ok" if await check_db_connection() else "unreachable"

    # Redis
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"unreachable: {e}"

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=3)
        qc.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"unreachable: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return APIResponse(data={"status": "healthy" if all_ok else "degraded", "checks": checks})


@router.post("/reindex-all", response_model=APIResponse[Dict[str, Any]], summary="Re-trigger ingestion for all failed documents")
async def reindex_all_failed(
    current_user: CurrentUser  = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Queue all documents with status='failed' for re-ingestion."""
    require_scope(current_user, "admin")

    result  = await db.execute(select(DocumentModel).where(DocumentModel.status == "failed"))
    failed  = result.scalars().all()
    queued  = []

    for doc in failed:
        doc.status = "pending"
        # TODO (Day 19): dispatch Celery task per doc
        queued.append(doc.id)

    return APIResponse(data={"queued_count": len(queued), "document_ids": queued})
