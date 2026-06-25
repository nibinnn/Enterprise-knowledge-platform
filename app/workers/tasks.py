"""
app/workers/tasks.py
─────────────────────────────────────────────────────────────────────────────
Celery task definitions for async document ingestion and evaluation.

ingest_document(doc_id, job_id, file_path)
    → parse → chunk → embed → upsert to Qdrant → update DB status

run_eval(run_id, questions)
    → for each question: ask() → compute RAGAS metrics → save EvalRun
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ── Ingestion task ─────────────────────────────────────────────────────────────

@celery_app.task(
    name="tasks.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def ingest_document(
    self,
    doc_id:    str,
    job_id:    str,
    file_path: str,
    metadata:  Optional[Dict[str, Any]] = None,
):
    """
    Full ingestion pipeline for one document.
    Runs inside a Celery worker process (sync).
    """
    from app.db.database import get_sync_session
    from app.db.models import DocumentModel, JobModel
    from sqlalchemy import select
    from datetime import datetime

    logger.info("[Task] Starting ingest: doc_id=%s file=%s", doc_id, file_path)

    with get_sync_session() as db:
        # Mark job as running
        job = db.execute(select(JobModel).where(JobModel.id == job_id)).scalar_one_or_none()
        if job:
            job.status = "running"
            job.celery_task_id = self.request.id
            db.commit()

    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Parse
        _update_job(job_id, "running", progress=10)
        from app.parsers import registry
        parser = registry.get_parser(path)
        doc    = parser.parse(path)

        # 2. Chunk
        _update_job(job_id, "running", progress=30)
        from app.chunkers.strategy_router import ChunkingStrategyRouter
        from app.config import get_settings
        s      = get_settings()
        router = ChunkingStrategyRouter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
        chunks = router.chunk(doc)

        # 3. Embed
        _update_job(job_id, "running", progress=55)
        import asyncio
        from app.core.base.embedder import EmbedderFactory
        embedder   = EmbedderFactory.get()
        texts      = [c.text for c in chunks]
        embeddings = asyncio.run(embedder.embed(texts))

        # 4. Build EmbeddedChunks
        from app.core.models.document import EmbeddedChunk
        embedded = []
        for chunk, vec in zip(chunks, embeddings):
            ec = EmbeddedChunk(**chunk.model_dump(), embedding=vec)
            embedded.append(ec)

        # 5. Upsert to Qdrant
        _update_job(job_id, "running", progress=75)
        from app.vector_store.qdrant_store import QdrantVectorStore
        vs       = QdrantVectorStore()
        upserted = asyncio.run(vs.upsert(embedded))

        # 6. Persist chunks to DB + update document status
        _update_job(job_id, "running", progress=90)
        from app.db.models import ChunkModel
        with get_sync_session() as db:
            for ec in embedded:
                m = ec.metadata
                db.add(ChunkModel(
                    id=ec.id,
                    doc_id=doc_id,
                    text=ec.text,
                    chunk_index=m.chunk_index,
                    char_count=m.char_count,
                    token_count=m.token_count,
                    chunking_strategy=m.chunking_strategy.value,
                    page_number=m.page_number,
                    section_heading=m.section_heading,
                    section_id=m.section_id,
                    is_embedded=True,
                    embedding_model=s.embedding_model,
                    embedded_at=datetime.utcnow(),
                    department=m.department,
                    doc_category=m.doc_category,
                ))

            document = db.execute(
                select(DocumentModel).where(DocumentModel.id == doc_id)
            ).scalar_one_or_none()
            if document:
                document.status      = "indexed"
                document.chunk_count = len(embedded)
                document.word_count  = doc.metadata.word_count
                document.page_count  = doc.metadata.page_count
                document.indexed_at  = datetime.utcnow()
                document.raw_text    = doc.raw_text[:50_000] if doc.raw_text else None  # cap at 50k chars
            db.commit()

        _update_job(job_id, "success", progress=100)
        logger.info("[Task] Ingestion complete: doc_id=%s chunks=%d", doc_id, len(embedded))
        return {"doc_id": doc_id, "chunks": len(embedded), "status": "indexed"}

    except Exception as exc:
        logger.error("[Task] Ingestion failed: doc_id=%s error=%s", doc_id, exc, exc_info=True)
        _update_job(job_id, "failed", error=str(exc))
        _mark_doc_failed(doc_id, str(exc))
        raise self.retry(exc=exc)


# ── Evaluation task ────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.run_eval", bind=True, max_retries=1)
def run_eval(self, run_id: str, questions: List[Dict[str, str]]):
    """
    Run RAGAS evaluation for a set of questions with ground-truth answers.
    questions: [{"question": ..., "ground_truth": ...}]
    """
    from app.db.database import get_sync_session
    from app.db.models import EvalRunModel
    from sqlalchemy import select
    import asyncio

    logger.info("[Task] Starting eval run: run_id=%s questions=%d", run_id, len(questions))

    results = []
    for q in questions:
        try:
            from app.rag.pipeline import RAGPipeline
            pipeline = RAGPipeline()
            answer   = asyncio.run(pipeline.run(q["question"]))
            results.append({
                "question":      q["question"],
                "ground_truth":  q.get("ground_truth", ""),
                "answer":        answer.answer_text,
                "contexts":      [c.excerpt for c in answer.citations],
            })
        except Exception as exc:
            logger.warning("[Eval] Question failed: %s", exc)

    # Compute RAGAS metrics
    metrics: Dict[str, Optional[float]] = {}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset  = Dataset.from_list(results)
        scores   = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        metrics  = {
            "faithfulness":      float(scores["faithfulness"]),
            "answer_relevance":  float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
            "context_recall":    float(scores["context_recall"]),
        }
    except Exception as exc:
        logger.warning("[Eval] RAGAS scoring failed: %s", exc)

    # Persist results
    with get_sync_session() as db:
        run = db.execute(select(EvalRunModel).where(EvalRunModel.id == run_id)).scalar_one_or_none()
        if run:
            run.faithfulness      = metrics.get("faithfulness")
            run.answer_relevance  = metrics.get("answer_relevance")
            run.context_precision = metrics.get("context_precision")
            run.context_recall    = metrics.get("context_recall")
            run.num_questions     = len(questions)
            run.results_json      = {"questions": results, "metrics": metrics}
            db.commit()

    logger.info("[Task] Eval complete: run_id=%s metrics=%s", run_id, metrics)
    return {"run_id": run_id, "metrics": metrics}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _update_job(job_id: str, status: str, progress: float = None, error: str = None):
    from app.db.database import get_sync_session
    from app.db.models import JobModel
    from sqlalchemy import select
    from datetime import datetime
    with get_sync_session() as db:
        job = db.execute(select(JobModel).where(JobModel.id == job_id)).scalar_one_or_none()
        if job:
            job.status       = status
            job.progress_pct = progress
            job.error        = error
            if status in ("success", "failed"):
                job.completed_at = datetime.utcnow()
            db.commit()


def _mark_doc_failed(doc_id: str, error: str):
    from app.db.database import get_sync_session
    from app.db.models import DocumentModel
    from sqlalchemy import select
    with get_sync_session() as db:
        doc = db.execute(select(DocumentModel).where(DocumentModel.id == doc_id)).scalar_one_or_none()
        if doc:
            doc.status        = "failed"
            doc.error_message = error[:1000]
            db.commit()
