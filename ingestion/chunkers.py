from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
)
from langchain_core.documents import Document
from config import ChunkingConfig, DocumentType
from utils.helpers import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Chunk pure-text / formula documents.
    Uses paragraph → sentence → word hierarchy to keep formulas intact.
    """

    def __init__(self, cfg: ChunkingConfig):
        self._splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=cfg.text_chunk_size,
            chunk_overlap=cfg.text_chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(docs)
        logger.debug("TextChunker: %d docs → %d chunks", len(docs), len(chunks))
        return chunks


class CodeChunker:
    """
    Chunk code-heavy documents using Python-aware splitting.
    Preserves function / class boundaries as much as possible.
    """

    def __init__(self, cfg: ChunkingConfig):
        self._splitter = PythonCodeTextSplitter(
            chunk_size=cfg.code_chunk_size,
            chunk_overlap=cfg.code_chunk_overlap,
            add_start_index=True,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(docs)
        logger.debug("CodeChunker: %d docs → %d chunks", len(docs), len(chunks))
        return chunks


class AdaptiveChunker:
    """
    Routes each document to the appropriate chunker based on doc_type metadata.
    Single entry-point used by the ingestion pipeline.
    """

    def __init__(self, cfg: ChunkingConfig):
        self._text_chunker = TextChunker(cfg)
        self._code_chunker = CodeChunker(cfg)

    def split(self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        """
        Returns (text_chunks, code_chunks).
        """
        text_docs = [d for d in docs if d.metadata.get("doc_type") != DocumentType.CODE.value]
        code_docs = [d for d in docs if d.metadata.get("doc_type") == DocumentType.CODE.value]

        text_chunks = self._text_chunker.split(text_docs) if text_docs else []
        code_chunks = self._code_chunker.split(code_docs) if code_docs else []

        logger.info(
            "Chunking complete → text: %d chunks, code: %d chunks",
            len(text_chunks), len(code_chunks),
        )
        return text_chunks, code_chunks
