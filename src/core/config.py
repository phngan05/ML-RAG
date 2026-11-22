from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QueryTranslationMethod(str, Enum):
    AUTO        = "auto"          # Let the router decide
    RAG_FUSION  = "rag_fusion"
    STEPBACK    = "stepback"
    HYDE        = "hyde"
    NONE        = "none"


class DocumentType(str, Enum):
    TEXT = "text"    # Pure text / formulas
    CODE = "code"    # Contains code


@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-3-small"
    chunk_overlap: int = 100


@dataclass
class VectorStoreConfig:
    text_index_name: str = "ml-text-knowledge"
    code_index_name: str = "ml-code-knowledge"
    dimension: int = 1536  # text-embedding-3-small output size


@dataclass
class LLMConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class ChunkingConfig:
    # Text / formula chunks
    text_chunk_size: int = 800
    text_chunk_overlap: int = 100
    # Code chunks
    code_chunk_size: int = 1200
    code_chunk_overlap: int = 150


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    rag_fusion_queries: int = 3       # How many sub-queries to generate
    self_rag_max_retries: int = 2     # Self-RAG retry limit
    retrieval_k: int = 5              # Docs per retrieval
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
