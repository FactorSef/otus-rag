"""Indexing pipeline: scan -> chunk -> embed -> upsert into Chroma."""

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_kb.indexing.chunkers import Chunker, search_text
from rag_kb.indexing.loaders import scan_folder, source_id
from rag_kb.retrieval.store import ChromaStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # (done, total, message)
MAX_REPORTED_SKIPS = 20


class IndexingError(ValueError):
    """Invalid indexing request (bad path, nothing to index)."""


@dataclass
class IndexReport:
    root: str
    pattern: str
    files_indexed: int = 0  # new or changed files (re-embedded)
    files_unchanged: int = 0
    files_removed: int = 0  # deleted from disk since the previous run
    chunks_written: int = 0
    total_files: int = 0
    total_chunks: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    skipped_count: int = 0
    skipped: list[str] = field(default_factory=list)
    index_reset: bool = False  # collection rebuilt because the embedding model changed
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Indexer:
    def __init__(
        self,
        store: ChromaStore,
        embeddings: Embeddings,
        chunker: Chunker,
        embedding_id: str,
        extensions: tuple[str, ...],
        max_file_size_kb: int,
        batch_size: int,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.chunker = chunker
        self.embedding_id = embedding_id
        self.extensions = extensions
        self.max_file_size_kb = max_file_size_kb
        self.batch_size = batch_size

    def run(
        self, root: Path, pattern: str = "**/*", progress: ProgressCallback | None = None
    ) -> IndexReport:
        started = time.perf_counter()
        root = root.expanduser()
        if not root.exists():
            raise IndexingError(f"Folder not found: {root}")
        if not root.is_dir():
            raise IndexingError(f"Not a folder: {root}")
        root = root.resolve()

        report = IndexReport(root=source_id(root), pattern=pattern)
        state = self.store.load_state()
        if self.store.count() and state.get("embedding_id") != self.embedding_id:
            logger.warning(
                "Embedding model changed (%s -> %s): rebuilding index",
                state.get("embedding_id"),
                self.embedding_id,
            )
            self.store.reset()
            report.index_reset = True

        scan = scan_folder(root, pattern, self.extensions, self.max_file_size_kb)
        report.skipped_count = len(scan.skipped)
        report.skipped = [f"{p}: {why}" for p, why in scan.skipped[:MAX_REPORTED_SKIPS]]
        if not scan.files:
            raise IndexingError(
                f"No supported files in {root} matching '{pattern}' "
                f"(supported: {', '.join(self.extensions)}; skipped {len(scan.skipped)})"
            )

        indexed = self.store.source_hashes()
        changed, chunks = [], []
        for file in scan.files:
            digest = hashlib.sha1(file.text.encode()).hexdigest()
            if indexed.get(file.source) == digest:
                report.files_unchanged += 1
                continue
            file_chunks = self.chunker.split(file)
            for chunk in file_chunks:
                chunk.metadata["file_hash"] = digest
            changed.append(file.source)
            chunks.extend(file_chunks)
            report.by_type[file.file_type] = report.by_type.get(file.file_type, 0) + 1

        stale = self._stale_sources(root, indexed)
        self.store.delete_sources(changed + stale)
        report.files_removed = len(stale)

        self._embed_and_store(chunks, progress)
        report.files_indexed = len(changed)
        report.chunks_written = len(chunks)

        hashes = self.store.source_hashes()
        report.total_files = len(hashes)
        report.total_chunks = self.store.count()
        report.duration_s = round(time.perf_counter() - started, 2)

        previous_roots = [] if report.index_reset else state.get("roots", [])
        roots = sorted({*previous_roots, report.root})
        self.store.save_state(
            {
                "embedding_id": self.embedding_id,
                "last_indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "roots": roots,
                "files_by_type": dict(Counter(s.rsplit(".", 1)[-1] for s in hashes)),
                "last_run": report.to_dict(),
            }
        )
        logger.info(
            "Indexed %s: %d changed, %d unchanged, %d removed, %d chunks in %.1fs",
            root,
            report.files_indexed,
            report.files_unchanged,
            report.files_removed,
            report.chunks_written,
            report.duration_s,
        )
        return report

    def _stale_sources(self, root: Path, indexed: dict[str, str]) -> list[str]:
        """Previously indexed files under ``root`` that no longer exist on disk."""
        root_id = source_id(root)
        prefix = "" if root_id == "." else root_id.rstrip("/") + "/"
        stale = []
        for source in indexed:
            if not source.startswith(prefix) or (not prefix and Path(source).is_absolute()):
                continue
            path = Path(source) if Path(source).is_absolute() else Path.cwd() / source
            if not path.exists():
                stale.append(source)
        return stale

    def _embed_and_store(self, chunks, progress: ProgressCallback | None) -> None:
        total = len(chunks)
        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self.embeddings.embed_documents([search_text(d) for d in batch])
            self.store.upsert(batch, vectors)
            done = start + len(batch)
            if progress:
                progress(done, total, f"Embedded {done}/{total} chunks")
