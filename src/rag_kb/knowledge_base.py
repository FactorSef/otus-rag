"""Facade tying together storage, indexing and hybrid search.

One instance lives for the lifetime of the server. The BM25 index is held in memory and
rebuilt from Chroma (the single source of truth) on start-up and after every indexing run.
Indexing runs in a background thread (see ``indexing/jobs.py``); searches keep working
on the previous data meanwhile.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from rag_kb.config import Settings
from rag_kb.indexing.chunkers import Chunker
from rag_kb.indexing.indexer import Indexer, IndexReport
from rag_kb.indexing.jobs import IndexingInProgressError, IndexJob, JobRunner
from rag_kb.retrieval.bm25 import BM25Index
from rag_kb.retrieval.embeddings import create_embeddings, embedding_id
from rag_kb.retrieval.hybrid import HybridSearch, SearchHit
from rag_kb.retrieval.store import ChromaStore

logger = logging.getLogger(__name__)


class EmptyIndexError(RuntimeError):
    """Search requested before anything was indexed."""


class IndexingFailedError(RuntimeError):
    """A background indexing job ended with an error."""


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
        self.jobs = JobRunner()
        self._bm25 = BM25Index([])
        self._bm25_generation = -1
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
        generation = self.store.generation
        self._bm25 = BM25Index(self.store.all_documents())  # atomic swap
        self._bm25_generation = generation
        logger.info("BM25 index: %d chunks", len(self._bm25))

    @property
    def is_empty(self) -> bool:
        return self.store.count() == 0

    def start_indexing(self, path: str | Path, pattern: str = "**/*") -> IndexJob:
        """Validate and plan synchronously, then embed in the background.

        Raises ``IndexingError`` (bad request) or ``IndexingInProgressError``.
        """
        if self.jobs.busy:  # fail fast, before planning against a half-written index
            raise IndexingInProgressError(
                "Another indexing job is running; check index_status and retry later."
            )
        plan = self.indexer.plan(Path(path), pattern or "**/*")
        return self.jobs.submit(plan, self.indexer.apply, on_finish=self._rebuild_bm25)

    def index_folder(self, path: str | Path, pattern: str = "**/*") -> IndexReport:
        """Index and wait for completion (scripts and tests)."""
        job = self.start_indexing(path, pattern)
        job.wait(None)
        if job.state == "failed" or job.report is None:
            raise IndexingFailedError(job.error or "indexing failed")
        return job.report

    def empty_message(self) -> str:
        """Why nothing can be searched yet, phrased for the host agent to relay."""
        job = self.jobs.current
        if job is not None and job.state == "running":
            percent = 100 * job.chunks_done / job.chunks_total if job.chunks_total else 0
            eta = f", about {round(job.eta_s / 60)} min left" if job.eta_s else ""
            return (
                f"The knowledge base is still being indexed in the background "
                f"({job.chunks_done}/{job.chunks_total} chunks, {percent:.0f}%{eta}). "
                "Try again when index_status shows state 'completed'."
            )
        return "The knowledge base is empty: call index_folder('./sample_docs') first."

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self.is_empty:
            raise EmptyIndexError(self.empty_message())
        stored = self.store.load_state().get("embedding_id")
        if stored and stored != self.embedding_id:
            # Query vectors of the configured model are not comparable with the index.
            raise EmptyIndexError(
                f"The index was built with '{stored}' but the server uses "
                f"'{self.embedding_id}': call index_folder to rebuild it."
            )
        if self._bm25_generation != self.store.generation:
            # The collection was reset by a running rebuild: drop BM25 hits of wiped chunks.
            self._rebuild_bm25()
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
        if self.jobs.current:
            result["indexing"] = self.jobs.current.to_dict()
        if chunks and stored_model and stored_model != self.embedding_id:
            result["warning"] = (
                f"Index was built with '{stored_model}' but the server is configured for "
                f"'{self.embedding_id}'. Re-run index_folder to rebuild it."
            )
        return result
