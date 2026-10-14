"""ChromaDB wrapper (vectors are computed by us, Chroma only stores and searches them)."""

import json
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STATE_FILE = "index_state.json"


class ChromaStore:
    """Thread-safe: indexing writes from a background thread while searches read.

    Every access to the collection goes through ``_lock``, so ``reset()`` (drop and
    recreate the collection) never runs in the middle of a query. ``generation`` grows
    on each reset and tells readers that caches built from the old data are stale.
    """

    def __init__(self, path: Path, collection: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.collection_name = collection
        self.generation = 0
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    # --- writes ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all chunks (used when the embedding model changes)."""
        with self._lock:
            self._client.delete_collection(self.collection_name)
            self._collection = self._open()
            self.generation += 1
            self.save_state({})

    def delete_sources(self, sources: list[str]) -> None:
        if sources:
            with self._lock:
                self._collection.delete(where={"source": {"$in": sources}})

    def upsert(self, docs: list[Document], vectors: list[list[float]], batch: int = 512) -> None:
        for i in range(0, len(docs), batch):
            part = docs[i : i + batch]
            with self._lock:
                self._collection.upsert(
                    ids=[d.id for d in part],
                    documents=[d.page_content for d in part],
                    metadatas=[d.metadata for d in part],
                    embeddings=vectors[i : i + batch],
                )

    # --- reads ----------------------------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            return self._collection.count()

    def source_hashes(self) -> dict[str, str]:
        """``{source: file_hash}`` for every *completely* indexed file.

        A file whose chunks were only partly written (indexing interrupted by an error or
        a restart) is left out, so the next run re-indexes it instead of skipping it.
        """
        with self._lock:
            got = self._collection.get(include=["metadatas"])
        counts = Counter(m["source"] for m in got["metadatas"])
        return {
            m["source"]: m.get("file_hash", "")
            for m in got["metadatas"]
            if counts[m["source"]] >= m.get("file_chunks", 0)
        }

    def all_documents(self) -> list[Document]:
        with self._lock:
            got = self._collection.get(include=["documents", "metadatas"])
        return [
            Document(id=i, page_content=d, metadata=m)
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        ]

    def get(self, ids: list[str]) -> list[Document]:
        """Chunks by id, in the order of ``ids``; missing ids are skipped."""
        if not ids:
            return []
        with self._lock:
            got = self._collection.get(ids=ids, include=["documents", "metadatas"])
        found = {
            i: Document(id=i, page_content=d, metadata=m)
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        }
        return [found[i] for i in ids if i in found]

    def query(self, vector: list[float], k: int) -> list[tuple[Document, float]]:
        """Nearest chunks with cosine similarity (1 - distance)."""
        with self._lock:
            total = self._collection.count()
            if total == 0:
                return []
            res = self._collection.query(
                query_embeddings=[vector],
                n_results=min(k, total),
                include=["documents", "metadatas", "distances"],
            )
        return [
            (Document(id=i, page_content=d, metadata=m), 1.0 - dist)
            for i, d, m, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]

    # --- index state (stats that survive restarts) -------------------------------------

    def load_state(self) -> dict[str, Any]:
        try:
            return json.loads((self.path / STATE_FILE).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_state(self, state: dict[str, Any]) -> None:
        tmp = self.path / f"{STATE_FILE}.tmp"
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path / STATE_FILE)
