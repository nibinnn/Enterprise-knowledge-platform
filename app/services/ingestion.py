"""
app/services/ingestion.py
Document ingestion: creates DB records and dispatches the Celery ingest task.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from app.config import get_settings

settings = get_settings()


class IngestionService:

    async def ingest_file(
        self,
        file_path: Path,
        original_filename: str,
        file_type: str,
        metadata: dict,
        db,
    ) -> dict:
        """
        Accept a file, create DB records, and dispatch the async ingestion task.
        Returns doc_id + job_id immediately; the Celery worker does the heavy lifting.
        """
        from app.db.models import DocumentModel, JobModel

        doc_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        doc = DocumentModel(
            id=doc_id,
            filename=original_filename,
            original_filename=original_filename,
            file_type=file_type,
            file_path=str(file_path),
            status="pending",
            title=metadata.get("title"),
            department=metadata.get("department"),
            doc_category=metadata.get("doc_category"),
            tags=metadata.get("tags", []),
            metadata_json=metadata.get("extra", {}),
        )
        db.add(doc)

        job = JobModel(
            id=job_id,
            doc_id=doc_id,
            job_type="ingest",
            status="pending",
        )
        db.add(job)
        await db.flush()

        from app.workers.celery_app import celery_app
        celery_app.send_task(
            "tasks.ingest_document",
            args=[doc_id, job_id, str(file_path)],
            kwargs={"metadata": metadata},
        )

        return {"document_id": doc_id, "job_id": job_id}

    async def get_status(self, document_id: str, db) -> Optional[dict]:
        from sqlalchemy import select
        from app.db.models import DocumentModel, JobModel

        result = await db.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            return None

        job_result = await db.execute(
            select(JobModel)
            .where(JobModel.doc_id == document_id)
            .order_by(JobModel.created_at.desc())
        )
        job = job_result.scalar_one_or_none()

        return {
            "document_id": document_id,
            "job_id": job.id if job else None,
            "status": doc.status,
            "progress_pct": job.progress_pct if job else None,
            "error": doc.error_message,
            "chunk_count": doc.chunk_count,
            "indexed_at": doc.indexed_at,
        }

    async def delete_document(self, document_id: str, db) -> bool:
        from sqlalchemy import select, delete
        from app.db.models import DocumentModel

        result = await db.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            return False

        await db.execute(
            delete(DocumentModel).where(DocumentModel.id == document_id)
        )

        from app.vector_store.qdrant_store import QdrantVectorStore
        vs = QdrantVectorStore()
        await vs.delete_document(document_id)

        return True
