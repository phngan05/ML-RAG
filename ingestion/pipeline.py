from pathlib import Path
from langchain_core.documents import Document
from ingestion.document_loader import DocumentLoader
from ingestion.chunkers import AdaptiveChunker
from retrieval.vector_stores import VectorStoreManager
from config import RAGConfig
from utils.helpers import get_logger, push_processed_file

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end pipeline: load → chunk → embed → store.
    """

    def __init__(self, cfg: RAGConfig, vector_store_manager: VectorStoreManager):
        self._loader = DocumentLoader()
        self._chunker = AdaptiveChunker(cfg.chunking)
        self._vsm = vector_store_manager

    def run(self, data_path: str | Path, processed_data_path: str | Path) -> dict[str, int]:
        logger.info("Starting ingestion from '%s'", data_path)
        raw_docs = self._loader.load(data_path)
        if not raw_docs:
            logger.warning("No documents found in '%s'", data_path)
            return {"text": 0, "code": 0}

        text_chunks, code_chunks = self._chunker.split(raw_docs)

        if text_chunks:
            self._vsm.add_documents("text", text_chunks)
        if code_chunks:
            self._vsm.add_documents("code", code_chunks)
            
        push_processed_file(data_path, processed_data_path)
        summary = {"text": len(text_chunks), "code": len(code_chunks)}
        logger.info("Ingestion complete: %s", summary)
        
        return summary
