"""Persistent SQLite state for asynchronous document ingestion."""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionStatusStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(
            path or os.environ.get(
                "INGESTION_STATUS_DB", "processed_data/ingestion_status.sqlite3"
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    document_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    error_message TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def submit(
        self,
        document_id: str,
        task_id: str,
        file_path: str,
        collection_name: str,
        metadata_json: str,
    ) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    document_id, task_id, file_path, collection_name, metadata,
                    status, error_message, chunk_count, created_at, started_at,
                    completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'PENDING', NULL, 0, ?, NULL, NULL, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    task_id=excluded.task_id, file_path=excluded.file_path,
                    collection_name=excluded.collection_name,
                    metadata=excluded.metadata, status='PENDING',
                    error_message=NULL, chunk_count=0, started_at=NULL,
                    completed_at=NULL, updated_at=excluded.updated_at
                """,
                (document_id, task_id, file_path, collection_name, metadata_json, now, now),
            )

    def update(self, document_id: str, status: str, **fields: Any) -> None:
        allowed = {"error_message", "chunk_count", "started_at", "completed_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["status"] = status
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {assignments} WHERE document_id=?",
                (*updates.values(), document_id),
            )

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, task_id, file_path, collection_name, status,
                       error_message, chunk_count, created_at, started_at,
                       completed_at, updated_at
                FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
