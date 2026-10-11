"""ChromaDB wrapper (vectors are computed by us, Chroma only stores and searches them)."""

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STATE_FILE = "index_state.json"


class ChromaStore:
    def __init__(self, path: Path, collection: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.collection_name = collection
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
        self._client.delete_collection(self.collection_name)
        self._collection = self._open()
        self.save_state({})

    def delete_sources(self, sources: list[str]) -> None:
        if sources:
            self._collection.delete(where={"source": {"$in": sources}})

    def upsert(self, docs: list[Document], vectors: list[list[float]], batch: int = 512) -> None:
        for i in range(0, len(docs), batch):
            part = docs[i : i + batch]
            self._collection.upsert(
                ids=[d.id for d in part],
                documents=[d.page_content for d in part],
                metadatas=[d.metadata for d in part],
                embeddings=vectors[i : i + batch],
            )

    # --- reads ----------------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def source_hashes(self) -> dict[str, str]:
        """``{source: file_hash}`` for every indexed file."""
        got = self._collection.get(include=["metadatas"])
        return {m["source"]: m.get("file_hash", "") for m in got["metadatas"]}

    def all_documents(self) -> list[Document]:
        got = self._collection.get(include=["documents", "metadatas"])
        return [
            Document(id=i, page_content=d, metadata=m)
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        ]

    def query(self, vector: list[float], k: int) -> list[tuple[Document, float]]:
        """Nearest chunks with cosine similarity (1 - distance)."""
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, self.count()),
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
