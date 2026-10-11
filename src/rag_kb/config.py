"""Application settings.

All tunables come from environment variables (or a ``.env`` file); the rest of
the code receives a ``Settings`` instance and never reads the environment itself.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openai"]
EmbeddingProvider = Literal["local", "ollama", "openai", "chroma"]

DEFAULT_EXTENSIONS = (".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `KEY=` in .env means "use the default"
        extra="ignore",
    )

    # --- MCP server ---
    server_host: str = "0.0.0.0"
    server_port: int = Field(8000, ge=1, le=65535)
    server_transport: Literal["http", "stdio"] = "http"
    log_level: str = "INFO"

    # --- LLM (external; only ask_question needs it) ---
    llm_provider: LLMProvider = "ollama"
    llm_base_url: str = "http://host.docker.internal:11434"
    llm_api_key: SecretStr | None = None
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(120.0, gt=0)

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_base_url: str | None = None  # falls back to llm_base_url
    embedding_api_key: SecretStr | None = None  # falls back to llm_api_key
    embedding_cache_dir: Path | None = None  # fastembed model cache (baked into the image)
    embedding_batch_size: int = Field(16, ge=1)

    # --- Storage ---
    chroma_path: Path = Path("./data/chroma")
    chroma_collection: str = "knowledge_base"

    # --- Indexing ---
    supported_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    max_file_size_kb: int = Field(1024, ge=1)
    chunk_size: int = Field(800, ge=100)
    chunk_overlap: int = Field(120, ge=0)
    # index_folder runs in the background; the tool call waits this long for it to finish
    # and otherwise returns immediately with a job to follow via index_status.
    index_wait_s: float = Field(5.0, ge=0)

    # --- Retrieval ---
    retrieve_k: int = Field(6, ge=1, le=50)
    max_top_k: int = Field(50, ge=1)
    fetch_k: int = Field(50, ge=1)  # depth of each list (BM25, vector) before fusion
    bm25_weight: float = Field(0.5, ge=0.0)
    vector_weight: float = Field(0.5, ge=0.0)
    rrf_c: int = Field(60, ge=1)

    # --- Corrective RAG graph ---
    min_relevant: int = Field(2, ge=1)
    max_retries: int = Field(2, ge=0)
    grade_max_chars: int = Field(1500, ge=200)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.bm25_weight + self.vector_weight == 0:
            raise ValueError("BM25_WEIGHT and VECTOR_WEIGHT cannot both be 0")
        if self.retrieve_k > self.max_top_k:
            raise ValueError("RETRIEVE_K cannot exceed MAX_TOP_K")
        if self.fetch_k < self.max_top_k:
            raise ValueError("FETCH_K must be at least MAX_TOP_K")
        return self

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> SecretStr | None:
        return self.embedding_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
