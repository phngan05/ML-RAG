from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from utils.helpers import get_logger, contains_code
from config import DocumentType

logger = get_logger(__name__)


class DocumentLoader:
    """Load PDF and Python source files, tagging each with its type."""

    SUPPORTED_SUFFIXES = {".pdf", ".py"}

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        return self._load_file(path)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _load_directory(self, directory: Path) -> list[Document]:
        docs: list[Document] = []
        for file in sorted(directory.rglob("*")):
            if file.suffix.lower() in self.SUPPORTED_SUFFIXES:
                docs.extend(self._load_file(file))
        logger.info("Loaded %d raw documents from '%s'", len(docs), directory)
        return docs

    def _load_file(self, path: Path) -> list[Document]:
        try:
            if path.suffix.lower() == ".pdf":
                return self._load_pdf(path)
            if path.suffix.lower() == ".py":
                return self._load_python(path)
        except Exception as exc:
            logger.warning("Skipping '%s': %s", path, exc)
        return []

    def _load_pdf(self, path: Path) -> list[Document]:
        pages = PyPDFLoader(str(path)).load()
        for doc in pages:
            doc_type = (
                DocumentType.CODE if contains_code(doc.page_content)
                else DocumentType.TEXT
            )
            doc.metadata.update(
                source=str(path),
                file_type="pdf",
                doc_type=doc_type.value,
            )
        logger.debug("PDF '%s' → %d pages", path.name, len(pages))
        return pages

    def _load_python(self, path: Path) -> list[Document]:
        docs = TextLoader(str(path), encoding="utf-8").load()
        for doc in docs:
            doc.metadata.update(
                source=str(path),
                file_type="py",
                doc_type=DocumentType.CODE.value,
            )
        return docs
