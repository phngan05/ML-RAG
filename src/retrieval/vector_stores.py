from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from src.core.config import RAGConfig
from src.utils.helpers import get_logger

logger = get_logger(__name__)

_STORE_KEYS = ("text", "code")


class VectorStoreManager:
    """
    Manages two separate Chroma collections:
      - "text"  → theory, definitions, formulas
      - "code"  → algorithms, implementations
    """

    def __init__(self, cfg: RAGConfig):
        self._cfg = cfg
        self._embedding = OpenAIEmbeddings(
            model=cfg.embedding.model_name,
            api_key=cfg.openai_api_key,
        )
        self.pc = Pinecone(api_key = cfg.pinecone_api_key)
        self._stores: dict[str, PineconeVectorStore] = {
            "text": self._build_store(cfg.vector_store.text_index_name),
            "code": self._build_store(cfg.vector_store.code_index_name),
        }

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def add_documents(
        self, store_key: str, docs: list[Document], ids: list[str] | None = None
    ) -> None:
        self._validate_key(store_key)
        self._stores[store_key].add_documents(docs, ids=ids)
        logger.info("Added %d docs to '%s' vector store", len(docs), store_key)

    def delete_document(self, document_id: str) -> None:
        """Remove all vectors for a document from both configured indexes."""
        for store_key, index_name in (
            ("text", self._cfg.vector_store.text_index_name),
            ("code", self._cfg.vector_store.code_index_name),
        ):
            self._validate_key(store_key)
            self.pc.Index(index_name).delete(filter={"document_id": document_id})
            logger.info("Deleted stale vectors for document %s from %s", document_id, store_key)

    def similarity_search(
        self, store_key: str, query: str, k: int = 5
    ) -> list[Document]:
        self._validate_key(store_key)
        return self._stores[store_key].similarity_search(query, k=k)

    def as_retriever(self, store_key: str, k: int = 5):
        self._validate_key(store_key)
        return self._stores[store_key].as_retriever(
            search_kwargs={"k": k}
        )

    def get_store(self, store_key: str) -> PineconeVectorStore:
        self._validate_key(store_key)
        return self._stores[store_key]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_store(self, index_name: str) -> PineconeVectorStore:
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name = index_name,
                dimension = self._cfg.vector_store.dimension,
                metric = "cosine",
                spec=ServerlessSpec(
                    cloud = "aws",
                    region = "us-east-1"
                )
            )
        
        index = self.pc.Index(index_name)
        return PineconeVectorStore(
            embedding = self._embedding,
            index = index
        )

    def _validate_key(self, key: str) -> None:
        if key not in _STORE_KEYS:
            raise ValueError(f"Unknown store key '{key}'. Choose from {_STORE_KEYS}.")
