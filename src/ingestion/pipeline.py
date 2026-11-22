from pathlib import Path
from datetime import datetime
import shutil

from langchain_core.documents import Document
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunkers import AdaptiveChunker
from src.retrieval.vector_stores import VectorStoreManager
from src.core.config import RAGConfig
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end pipeline: load → chunk → embed → store.
    """

    def __init__(self, cfg: RAGConfig, vector_store_manager: VectorStoreManager):
        self._loader = DocumentLoader()
        self._chunker = AdaptiveChunker(cfg.chunking)
        self._vsm = vector_store_manager

    def ingest_one(
        self,
        file: str | Path,
        document_id: str,
        metadata: dict | None = None,
        batch_size: int = 64,
    ) -> int:
        """Idempotently upsert one file's chunks, embedding bounded batches."""
        path = Path(file)
        if path.suffix.lower() not in self._loader.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {path.suffix}")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        raw_docs = self._loader.load(path)
        if not raw_docs:
            raise ValueError(f"No documents were loaded from '{path}'")
        for doc in raw_docs:
            doc.metadata.update(metadata or {})
            doc.metadata["document_id"] = document_id
            doc.metadata["source"] = str(path)

        text_chunks, code_chunks = self._chunker.split(raw_docs)
        chunks = [
            ("text", text_chunks),
            ("code", code_chunks),
        ]
        total = len(text_chunks) + len(code_chunks)
        if total == 0:
            raise ValueError(f"No chunks were created from '{path}'")

        self._vsm.delete_document(document_id)
        for store_key, store_chunks in chunks:
            for offset in range(0, len(store_chunks), batch_size):
                batch = store_chunks[offset:offset + batch_size]
                ids = [
                    f"{document_id}-{store_key}-{offset + index}"
                    for index in range(len(batch))
                ]
                self._vsm.add_documents(store_key, batch, ids=ids)
                logger.info(
                    "Document '%s': upserted %d/%d %s chunks",
                    path.name,
                    min(offset + len(batch), len(store_chunks)),
                    len(store_chunks),
                    store_key,
                )
        return total

    def run(self, data_path: str | Path, processed_data_path: str | Path) -> dict[str, int]:
        source = Path(data_path)
        destination = Path(processed_data_path)
        files = (
            [source]
            if source.is_file()
            else sorted(
                file
                for file in source.rglob("*")
                if file.is_file()
                and file.suffix.lower() in self._loader.SUPPORTED_SUFFIXES
            )
        )
        summary = {"text": 0, "code": 0, "files_processed": 0, "files_failed": 0}
        logger.info("Starting ingestion of %d file(s) from '%s'", len(files), source)

        for index, file in enumerate(files, start=1):
            logger.info("File %d/%d: loading '%s'", index, len(files), file)
            try:
                raw_docs = self._loader.load(file)
                if not raw_docs:
                    raise ValueError("Loader returned no documents")
                logger.info("File %d/%d: loaded %d document(s)", index, len(files), len(raw_docs))

                text_chunks, code_chunks = self._chunker.split(raw_docs)
                logger.info(
                    "File %d/%d: created %d text and %d code chunk(s)",
                    index,
                    len(files),
                    len(text_chunks),
                    len(code_chunks),
                )

                if text_chunks:
                    self._vsm.add_documents("text", text_chunks)
                    logger.info("File %d/%d: upserted text chunks to Pinecone", index, len(files))
                if code_chunks:
                    self._vsm.add_documents("code", code_chunks)
                    logger.info("File %d/%d: upserted code chunks to Pinecone", index, len(files))

                self._move_processed_file(file, destination)
                summary["text"] += len(text_chunks)
                summary["code"] += len(code_chunks)
                summary["files_processed"] += 1
                logger.info("File %d/%d: completed '%s'", index, len(files), file)
            except Exception:
                summary["files_failed"] += 1
                logger.exception("File %d/%d: failed '%s'", index, len(files), file)

        logger.info("Ingestion complete: %s", summary)
        return summary

    @staticmethod
    def _move_processed_file(file: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / file.name
        if target.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = destination / f"{file.stem}_{timestamp}{file.suffix}"
        shutil.move(str(file), str(target))
