"""Embedding providers, selected by ``EMBEDDING_PROVIDER``.

All providers are exposed through LangChain's ``Embeddings`` interface, so the rest of
the code does not depend on the choice.
"""

import logging
import re
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from rag_kb.config import Settings

logger = logging.getLogger(__name__)

# e5-large, multilingual-e5-base, intfloat/e5-small-v2, ollama "...-e5-large:latest"
E5_NAME_RE = re.compile(r"(^|[-_/])e5([-_:]|$)")


class PrefixedEmbeddings(Embeddings):
    """Adds instruction prefixes required by some models (E5: ``query:`` / ``passage:``)."""

    def __init__(self, inner: Embeddings, query_prefix: str, passage_prefix: str) -> None:
        self.inner = inner
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings via fastembed; the model is loaded on first use."""

    def __init__(self, model_name: str, cache_dir: str | None, batch_size: int) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ChromaDefaultEmbeddings(Embeddings):
    """ChromaDB's built-in embedding function (all-MiniLM-L6-v2, English only)."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def instruction_prefixes(model_name: str) -> tuple[str, str]:
    """``(query_prefix, passage_prefix)`` the model was trained with."""
    if E5_NAME_RE.search(model_name.lower()):
        return "query: ", "passage: "
    return "", ""


def embedding_id(settings: Settings) -> str:
    """Identifies the vector space; vectors of different models are not comparable."""
    if settings.embedding_provider == "chroma":
        return "chroma:default"
    return f"{settings.embedding_provider}:{settings.embedding_model}"


def create_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    if provider == "local":
        cache = str(settings.embedding_cache_dir) if settings.embedding_cache_dir else None
        inner: Embeddings = FastEmbedEmbeddings(
            settings.embedding_model, cache, settings.embedding_batch_size
        )
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        inner = OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.effective_embedding_base_url
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        key = settings.effective_embedding_api_key
        inner = OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.effective_embedding_base_url,
            api_key=key.get_secret_value() if key else "not-needed",
            check_embedding_ctx_length=False,  # tiktoken-based check breaks non-OpenAI models
            chunk_size=settings.embedding_batch_size,
        )
    elif provider == "chroma":
        return ChromaDefaultEmbeddings()
    else:  # pragma: no cover - guarded by Settings validation
        raise ValueError(f"Unknown embedding provider: {provider}")

    query_prefix, passage_prefix = instruction_prefixes(settings.embedding_model)
    if query_prefix or passage_prefix:
        return PrefixedEmbeddings(inner, query_prefix, passage_prefix)
    return inner


def batched(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
