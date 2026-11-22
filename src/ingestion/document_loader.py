import threading
from pathlib import Path
import fitz
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import NotebookLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.utils.helpers import get_logger, contains_code
from src.core.config import DocumentType

logger = get_logger(__name__)
_MUPDF_LOCK = threading.Lock()


class DocumentLoader:
    """Load PDF and Python source files, tagging each with its type."""

    SUPPORTED_SUFFIXES = {".pdf", ".py", ".ipynb"}

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        return self._load_file(path)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _load_file(self, path: Path) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        if path.suffix.lower() == ".py":
            return self._load_python(path)
        if path.suffix.lower() == ".ipynb":
            return self._load_ipynb(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load_pdf(self, path: Path) -> list[Document]:
        pages: list[Document] = []
        pymupdf_error: Exception | None = None
        mupdf_diagnostics = ""
        with _MUPDF_LOCK:
            display_errors = fitz.TOOLS.mupdf_display_errors()
            fitz.TOOLS.reset_mupdf_warnings()
            fitz.TOOLS.mupdf_display_errors(False)
            try:
                pages = PyMuPDF4LLMLoader(str(path)).load()
            except Exception as exc:
                pymupdf_error = exc
            finally:
                mupdf_diagnostics = fitz.TOOLS.mupdf_warnings()
                fitz.TOOLS.mupdf_display_errors(display_errors)

        has_syntax_error = "syntax error" in mupdf_diagnostics.lower()
        if (
            pymupdf_error
            or has_syntax_error
            or not any(doc.page_content.strip() for doc in pages)
        ):
            reason = str(pymupdf_error or mupdf_diagnostics or "no extractable text")
            logger.warning(
                "PyMuPDF could not fully read '%s' (%s); retrying with pypdf",
                path,
                reason,
            )
            try:
                pages = PyPDFLoader(str(path)).load()
            except Exception as pypdf_error:
                raise RuntimeError(
                    f"PDF extraction failed with both PyMuPDF and pypdf: {path}"
                ) from pypdf_error

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
    
    
    def _load_ipynb(self, path: Path) -> list[Document]:
        docs = NotebookLoader(str(path), include_outputs=False).load()
        for doc in docs:
            doc_type = (
                DocumentType.CODE if contains_code(doc.page_content)
                else DocumentType.TEXT
            )
            doc.metadata.update(
                source=str(path),
                file_type="ipynb",
                doc_type=doc_type.value,
            )
        return docs