"""Facade tying together storage, indexing and hybrid search.

One instance lives for the lifetime of the server. The BM25 index is held in memory and
rebuilt from Chroma (the single source of truth) on start-up and after every indexing run.
"""

import logging
import threading
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from rag_kb.config import Settings
from rag_kb.indexing.chunkers import Chunker
from rag_kb.indexing.indexer import Indexer, IndexReport, ProgressCallback
from rag_kb.retrieval.bm25 import BM25Index
from rag_kb.retrieval.embeddings import create_embeddings, embedding_id
from rag_kb.retrieval.hybrid import HybridSearch, SearchHit
from rag_kb.retrieval.store import ChromaStore

logger = logging.getLogger(__name__)


class EmptyIndexError(RuntimeError):
    """Search requested before anything was indexed."""


class KnowledgeBase:
    def __init__(self, settings: Settings, embeddings: Embeddings | None = None) -> None:
        self.settings = settings
        self.embedding_id = embedding_id(settings)
        self.embeddings = embeddings or create_embeddings(settings)
        self.store = ChromaStore(settings.chroma_path, settings.chroma_collection)
        self.indexer = Indexer(
            store=self.store,
            embeddings=self.embeddings,
            chunker=Chunker(settings.chunk_size, settings.chunk_overlap),
            embedding_id=self.embedding_id,
            extensions=settings.supported_extensions,
            max_file_size_kb=settings.max_file_size_kb,
            batch_size=settings.embedding_batch_size,
        )
        self._lock = threading.Lock()  # one indexing run at a time
        self._bm25 = BM25Index([])
        self.search_engine = HybridSearch(
            store=self.store,
            embeddings=self.embeddings,
            bm25_provider=lambda: self._bm25,
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
            rrf_c=settings.rrf_c,
            fetch_k=settings.fetch_k,
        )
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Index(self.store.all_documents())  # atomic swap
        logger.info("BM25 index: %d chunks", len(self._bm25))

    @property
    def is_empty(self) -> bool:
        return self.store.count() == 0

    def index_folder(
        self, path: str | Path, pattern: str = "**/*", progress: ProgressCallback | None = None
    ) -> IndexReport:
        with self._lock:
            report = self.indexer.run(Path(path), pattern or "**/*", progress)
            self._rebuild_bm25()
            return report

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self.is_empty:
            raise EmptyIndexError("The knowledge base is empty: call index_folder first.")
        return self.search_engine.search(query, k)

    def status(self) -> dict[str, Any]:
        state = self.store.load_state()
        stored_model = state.get("embedding_id")
        chunks = self.store.count()
        result: dict[str, Any] = {
            "empty": chunks == 0,
            "files": len(self.store.source_hashes()) if chunks else 0,
            "chunks": chunks,
            "files_by_type": state.get("files_by_type", {}) if chunks else {},
            "last_indexed_at": state.get("last_indexed_at") if chunks else None,
            "indexed_roots": state.get("roots", []) if chunks else [],
            "embedding_model": self.embedding_id,
        }
        if chunks and stored_model and stored_model != self.embedding_id:
            result["warning"] = (
                f"Index was built with '{stored_model}' but the server is configured for "
                f"'{self.embedding_id}'. Re-run index_folder to rebuild it."
            )
        return result
