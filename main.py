import os
from config import RAGConfig, QueryTranslationMethod
from ingestion.pipeline import IngestionPipeline
from retrieval.vector_stores import VectorStoreManager
from retrieval.retriever import Retriever
from generation.self_rag import SelfRAG
from langchain_groq import ChatGroq
from utils.helpers import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)

load_dotenv()

class MLKnowledgeRAG:
    """
    Top-level facade. Wires all components together.

    Usage:
        rag = MLKnowledgeRAG(groq_api_key="sk-...")
        rag.ingest("./data")
        result = rag.query("How does backpropagation work?")
        print(result["answer"])
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        cfg: RAGConfig | None = None,
    ):
        self._cfg = cfg or RAGConfig()
        self._cfg.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._cfg.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY", "")
        self._llm = ChatGroq(
            model=self._cfg.llm.model_name,
            temperature=self._cfg.llm.temperature,
            max_tokens=self._cfg.llm.max_tokens,
            api_key=self._cfg.groq_api_key,
        )

        self._vsm = VectorStoreManager(self._cfg)
        self._retriever = Retriever(self._cfg, self._vsm, self._llm)
        self._self_rag = SelfRAG(self._cfg, self._retriever, self._llm)

    # ------------------------------------------------------------------ #

    def ingest(self, data_path: str, processed_data_path: str) -> dict[str, int]:
        """Load, chunk, embed and store all documents at data_path."""
        pipeline = IngestionPipeline(self._cfg, self._vsm)
        return pipeline.run(data_path, processed_data_path)

    def query(
        self,
        question: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> dict:
        """Ask a question; returns answer, sources, and grading metadata."""
        return self._self_rag.answer(question, method=method)

    def print_result(self, result: dict) -> None:
        print("\n" + "=" * 70)
        print(f"ANSWER\n{'=' * 70}")
        print(result["answer"])
        print(f"\n{'=' * 70}")
        print(f"Translation method : {result['translation_method']}")
        print(f"Attempts           : {result['attempts']}")
        if "warning" in result:
            print(f"Warning         : {result['warning']}")
        print(f"\nSOURCES ({len(result['sources'])} docs)")
        print("-" * 70)
        for i, doc in enumerate(result["sources"], 1):
            src = doc.metadata.get("source", "unknown")
            dtype = doc.metadata.get("doc_type", "?")
            print(f"[{i}] ({dtype}) {src}")
            print(f"    {doc.page_content[:120].strip()}…")
        print("=" * 70 + "\n")


# ======================================================================== #
# Entry point
# ======================================================================== #

if __name__ == "__main__":
    # rag = MLKnowledgeRAG(groq_api_key=os.environ.get("groq_API_KEY"))
    
    # # ── Ingest documents ──────────────────────────────────────────────────
    # stats = rag.ingest("./data", "./processed_data")
    # print(f"Ingestion complete: {stats}")

    # # ── Example queries with explicit methods ─────────────────────────────
    # question = input("Enter question: ")
    # result = rag.query(question)
    # rag.print_result(result)
    print("Hello World!")
