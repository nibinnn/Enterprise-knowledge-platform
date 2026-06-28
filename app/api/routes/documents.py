"""app/api/routes/documents.py — document ingestion and management endpoints."""
from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException,
    Query, UploadFile, status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db, get_ingestion_service
from app.api.schemas.common import APIResponse, PaginatedResponse, PaginationMeta
from app.api.schemas.documents import (
    DocumentFilterParams, DocumentListOut,
    DocumentOut, DocumentStatusOut, DocumentUploadResponse,
)
from app.api.schemas.auth import CurrentUser
from app.config import get_settings
from app.db.models import DocumentModel
from app.services.ingestion import IngestionService

router  = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=APIResponse[DocumentUploadResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
)
async def upload_document(
    file:         UploadFile = File(...),
    title:        Optional[str] = Form(default=None),
    department:   Optional[str] = Form(default=None),
    doc_category: Optional[str] = Form(default=None),
    tags:         Optional[str] = Form(default=None),  # comma-separated
    current_user: CurrentUser = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    svc:          IngestionService = Depends(get_ingestion_service),
):
    """
    Upload a document (PDF, DOCX, TXT, MD, HTML) for asynchronous ingestion.
    Returns immediately with document_id + job_id for status polling.
    """
    _validate_file(file)

    # Save to upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / safe_name

    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )
    file_path.write_bytes(contents)

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    ext      = Path(file.filename).suffix.lstrip(".").lower()

    ids = await svc.ingest_file(
        file_path=file_path,
        original_filename=file.filename,
        file_type=ext,
        metadata={"title": title, "department": department,
                  "doc_category": doc_category, "tags": tag_list},
        db=db,
    )

    return APIResponse(
        data=DocumentUploadResponse(
            document_id=ids["document_id"],
            job_id=ids["job_id"],
            filename=file.filename,
            file_type=ext,
        )
    )


# ── List ──────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=PaginatedResponse[DocumentListOut],
    summary="List all documents",
)
async def list_documents(
    page:         int = Query(default=1, ge=1),
    page_size:    int = Query(default=20, ge=1, le=100),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    file_type:    Optional[str] = Query(default=None),
    department:   Optional[str] = Query(default=None),
    search:       Optional[str] = Query(default=None, description="Search by filename/title"),
    current_user: CurrentUser = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Return a paginated list of all documents with optional filters."""
    q = select(DocumentModel)

    if status_filter:
        q = q.where(DocumentModel.status == status_filter)
    if file_type:
        q = q.where(DocumentModel.file_type == file_type)
    if department:
        q = q.where(DocumentModel.department == department)
    if search:
        pattern = f"%{search}%"
        q = q.where(
            DocumentModel.filename.ilike(pattern)
            | DocumentModel.title.ilike(pattern)
        )

    # Count total
    count_q  = select(func.count()).select_from(q.subquery())
    total    = (await db.execute(count_q)).scalar_one()

    # Paginated results
    rows = (await db.execute(
        q.order_by(DocumentModel.created_at.desc())
         .offset((page - 1) * page_size)
         .limit(page_size)
    )).scalars().all()

    total_pages = max(1, -(-total // page_size))   # ceil division

    return PaginatedResponse(
        data=[DocumentListOut.model_validate(r) for r in rows],
        meta=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        ),
    )


# ── Get ───────────────────────────────────────────────────────────────────────

@router.get(
    "/{document_id}",
    response_model=APIResponse[DocumentOut],
    summary="Get document details",
)
async def get_document(
    document_id:  str,
    current_user: CurrentUser = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    doc = await _get_or_404(document_id, db)
    return APIResponse(data=DocumentOut.model_validate(doc))


# ── Status ────────────────────────────────────────────────────────────────────

@router.get(
    "/{document_id}/status",
    response_model=APIResponse[DocumentStatusOut],
    summary="Poll ingestion status",
)
async def get_document_status(
    document_id:  str,
    current_user: CurrentUser = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    svc:          IngestionService = Depends(get_ingestion_service),
):
    """Poll this endpoint after upload to track ingestion progress."""
    result = await svc.get_status(document_id, db)
    if not result:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return APIResponse(data=DocumentStatusOut(**result))


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id:  str,
    current_user: CurrentUser = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    svc:          IngestionService = Depends(get_ingestion_service),
):
    deleted = await svc.delete_document(document_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return None


# ── Retry / Re-index ──────────────────────────────────────────────────────────

@router.post(
    "/{document_id}/retry",
    response_model=APIResponse[DocumentUploadResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry a failed or stuck ingestion job",
)
async def retry_ingestion(
    document_id:  str,
    current_user: CurrentUser = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    svc:          IngestionService = Depends(get_ingestion_service),
):
    """
    Re-dispatch the Celery ingestion task for an existing document.
    Use this when a job failed, got stuck in 'pending', or needs to be re-run
    after a config change. Creates a new job record; the document_id is preserved.
    """
    doc = await _get_or_404(document_id, db)

    if doc.status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion is already running for this document.",
        )

    try:
        ids = await svc.retry_ingest(document_id, db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    if not ids:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    return APIResponse(
        data=DocumentUploadResponse(
            document_id=ids["document_id"],
            job_id=ids["job_id"],
            filename=doc.original_filename,
            file_type=doc.file_type,
            message="Ingestion job re-queued.",
        )
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_or_404(document_id: str, db: AsyncSession) -> DocumentModel:
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return doc


def _validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extension_list:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{ext}' not supported. "
                   f"Allowed: {settings.allowed_extensions}",
        )
