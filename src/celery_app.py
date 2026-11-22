"""Celery application configuration for background workers."""

import os

from celery import Celery
from dotenv import load_dotenv
from kombu import Queue

load_dotenv()

celery_app = Celery(
    "ml_rag",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=["src.ingestion.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=60 * 30,
    task_time_limit=60 * 35,
    task_default_queue="default",
    task_queues=(Queue("default"), Queue("ingestion")),
    task_routes={"src.ingestion.tasks.ingest_document_task": {"queue": "ingestion"}},
)
