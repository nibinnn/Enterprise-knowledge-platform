"""
app/workers/celery_app.py
─────────────────────────────────────────────────────────────────────────────
Celery application definition. Import this anywhere you need to dispatch tasks.
Workers are started with:
    celery -A app.workers.celery_app worker --loglevel=info
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ekip",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                  # re-queue on worker crash
    worker_prefetch_multiplier=1,         # one task at a time per worker
    task_routes={
        "tasks.ingest_document":  {"queue": "ingestion"},
        "tasks.run_eval":         {"queue": "eval"},
    },
)
