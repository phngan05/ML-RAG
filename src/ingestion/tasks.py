"""Celery tasks and enqueue helpers for document ingestion."""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from celery.exceptions import SoftTimeLimitExceeded
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from src.celery_app import celery_app
from src.core.config import RAGConfig
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.status_store import IngestionStatusStore
from src.retrieval.vector_stores import VectorStoreManager
from src.utils.helpers import get_logger

logger = get_logger(__name__)

TRANSIENT_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    RequestsConnectionError,
    RequestsTimeout,
    TimeoutError,
    ConnectionError,
)


def document_identifier(file_path: str | Path) -> str:
    return hashlib.sha256(str(Path(file_path).resolve()).encode("utf-8")).hexdigest()


def enqueue_document(
    file_path: str | Path,
    processed_data_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")
    document_id = document_identifier(path)
    task_id = str(uuid4())
    serialized_metadata = json.dumps(metadata or {}, ensure_ascii=True)
    IngestionStatusStore().submit(
        document_id,
        task_id,
        str(path),
        "text+code",
        serialized_metadata,
    )
    try:
        ingest_document_task.apply_async(
            args=[document_id, str(path), str(Path(processed_data_path).resolve()), serialized_metadata],
            task_id=task_id,
            queue="ingestion",
        )
    except Exception as exc:
        IngestionStatusStore().update(document_id, "FAILED", error_message=str(exc))
        raise
    return {"document_id": document_id, "task_id": task_id}


@celery_app.task(
    bind=True,
    name="src.ingestion.tasks.ingest_document_task",
    autoretry_for=TRANSIENT_ERRORS,
    retry_backoff=True,
    max_retries=3,
    acks_late=True,
)
def ingest_document_task(
    self,
    document_id: str,
    file_path: str,
    processed_data_path: str,
    metadata_json: str = "{}",
) -> dict[str, Any]:
    status = IngestionStatusStore()
    status.update(
        document_id,
        "PROCESSING",
        started_at=datetime.now(timezone.utc).isoformat(),
        error_message=None,
        completed_at=None,
    )
    try:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() not in {".pdf", ".py", ".ipynb"}:
            raise ValueError(f"Unsupported document type: {path.suffix}")

        config = RAGConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        )
        metadata = json.loads(metadata_json)
        vector_stores = VectorStoreManager(config)
        chunks = IngestionPipeline(config, vector_stores).ingest_one(
            path,
            document_id=document_id,
            metadata=metadata,
            batch_size=64,
        )

        destination = Path(processed_data_path)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / path.name
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = destination / f"{path.stem}_{stamp}{path.suffix}"
        shutil.move(str(path), str(target))

        status.update(
            document_id,
            "COMPLETED",
            chunk_count=chunks,
            error_message=None,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        return {"document_id": document_id, "status": "COMPLETED", "chunk_count": chunks}
    except TRANSIENT_ERRORS as exc:
        retrying = self.request.retries < self.max_retries
        status.update(
            document_id,
            "PENDING" if retrying else "FAILED",
            error_message=str(exc),
            completed_at=None if retrying else datetime.now(timezone.utc).isoformat(),
        )
        raise
    except SoftTimeLimitExceeded as exc:
        status.update(
            document_id,
            "FAILED",
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        raise
    except Exception as exc:
        status.update(
            document_id,
            "FAILED",
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.exception("Ingestion failed for document %s", document_id)
        raise
