"""Backend service layer for the Streamlit UI.

The UI only talks to :class:`RAGBackend`. All RAG wiring, configuration and
async plumbing lives here so the frontend stays a thin presentation layer
and the core Self-RAG algorithm is left untouched.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from src.core.config import LLMConfig, QueryTranslationMethod, RAGConfig
from src.main import MLKnowledgeRAG
from src.utils.helpers import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_PATH = "data"
DEFAULT_PROCESSED_PATH = "processed_data"

_T = TypeVar("_T")


class _AsyncLoopThread:
    """
    A single long-lived event loop.

    LLM/HTTP async clients bind their connection pools to the loop they are
    first used on, so every query must run on the same loop. Streamlit reruns
    the script per interaction, hence the process-wide loop instead of one
    loop per query.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="rag-async-loop", daemon=True
        )
        self._thread.start()

    def run(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()


_runner: _AsyncLoopThread | None = None
_runner_lock = threading.Lock()


def run_sync(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Resolve a coroutine from synchronous code on the shared event loop."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = _AsyncLoopThread()
    return _runner.run(coroutine)


@dataclass
class BackendSettings:
    """User-tunable knobs exposed by the UI."""

    openai_api_key: str = ""
    pinecone_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    retrieval_k: int = 5
    self_rag_max_retries: int = 2
    rag_fusion_queries: int = 3

    def to_config(self) -> RAGConfig:
        return RAGConfig(
            llm=LLMConfig(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            retrieval_k=self.retrieval_k,
            self_rag_max_retries=self.self_rag_max_retries,
            rag_fusion_queries=self.rag_fusion_queries,
            openai_api_key=self.openai_api_key,
            pinecone_api_key=self.pinecone_api_key,
        )

    def missing_credentials(self) -> list[str]:
        return [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                ("PINECONE_API_KEY", self.pinecone_api_key),
            )
            if not value
        ]


def settings_from_env() -> BackendSettings:
    """Defaults for the UI, taken from `.env`/environment via the RAGConfig."""
    cfg = RAGConfig()
    return BackendSettings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "") or cfg.openai_api_key or "",
        pinecone_api_key=os.environ.get("PINECONE_API_KEY", "") or cfg.pinecone_api_key or "",
        model_name=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        retrieval_k=cfg.retrieval_k,
        self_rag_max_retries=cfg.self_rag_max_retries,
        rag_fusion_queries=cfg.rag_fusion_queries,
    )


class RAGBackend:
    """Lifecycle + configuration facade used by the Streamlit frontend."""

    def __init__(self, settings: BackendSettings | None = None):
        self._settings = settings or settings_from_env()
        self._require_credentials(self._settings)

        self._cfg = self._settings.to_config()
        self._rag = MLKnowledgeRAG(
            openai_api_key=self._cfg.openai_api_key,
            pinecone_api_key=self._cfg.pinecone_api_key,
            cfg=self._cfg,
        )

    @property
    def settings(self) -> BackendSettings:
        return self._settings

    def apply_settings(self, settings: BackendSettings) -> None:
        """
        Apply UI changes without reloading embeddings or vector stores.

        Retrieval settings are read from the config on every call, so only the
        LLM pipeline needs rebuilding when model parameters change.
        """
        self._require_credentials(settings)

        if (
            settings.openai_api_key,
            settings.pinecone_api_key,
        ) != (
            self._settings.openai_api_key,
            self._settings.pinecone_api_key,
        ):
            raise ValueError(
                "Credentials changed; build a new RAGBackend instead of reusing this one."
            )

        llm_changed = (
            settings.model_name,
            settings.temperature,
            settings.max_tokens,
        ) != (
            self._settings.model_name,
            self._settings.temperature,
            self._settings.max_tokens,
        )

        self._settings = settings
        self._cfg.retrieval_k = settings.retrieval_k
        self._cfg.self_rag_max_retries = settings.self_rag_max_retries
        self._cfg.rag_fusion_queries = settings.rag_fusion_queries

        if llm_changed:
            logger.info("Switching chat model to '%s'", settings.model_name)
            self._rag.reconfigure_llm(
                model_name=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

    def ingest(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        processed_data_path: str | Path = DEFAULT_PROCESSED_PATH,
    ) -> dict[str, int]:
        """Chunk and upsert every supported document under `data_path`."""
        source = Path(data_path)
        if not source.exists():
            raise FileNotFoundError(f"Data path not found: {source}")

        stats = self._rag.ingest(str(source), str(processed_data_path))
        logger.info("Ingestion finished: %s", stats)
        return stats

    def ask(
        self,
        question: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> dict[str, Any]:
        """Run one Self-RAG query and return its answer payload."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return run_sync(self._rag.query(question, method=method))

    @staticmethod
    def _require_credentials(settings: BackendSettings) -> None:
        missing = settings.missing_credentials()
        if missing:
            raise ValueError("Missing required credentials: " + ", ".join(missing))
