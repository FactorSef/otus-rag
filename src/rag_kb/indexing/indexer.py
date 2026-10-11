"""Indexing pipeline: scan -> chunk -> embed -> upsert into Chroma."""

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_kb.indexing.chunkers import Chunker, search_text
from rag_kb.indexing.loaders import scan_folder, source_id
from rag_kb.retrieval.embeddings import batched
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


@dataclass
class IndexPlan:
    """What an indexing run will do; produced quickly, before any embedding work."""

    root: Path
    report: IndexReport
    previous_roots: list[str]
    changed: list[str] = field(default_factory=list)  # sources to (re-)embed
    empty: dict[str, str] = field(default_factory=dict)  # files without chunks: {source: hash}
    previous_empty: dict[str, str] = field(default_factory=dict)
    rechunk: bool = False  # chunking settings changed: every file is re-chunked
    stale: list[str] = field(default_factory=list)  # sources gone from disk
    chunks: list[Document] = field(default_factory=list)


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

    def plan(self, root: Path, pattern: str = "**/*") -> IndexPlan:
        """Validate the request, scan and chunk files. Fast and read-only.

        Raises ``IndexingError`` for a bad path or when nothing can be indexed.
        """
        root = root.expanduser()
        if not root.exists():
            raise IndexingError(f"Folder not found: {root}")
        if not root.is_dir():
            raise IndexingError(f"Not a folder: {root}")
        root = root.resolve()
        _check_pattern(pattern)

        scan = scan_folder(root, pattern, self.extensions, self.max_file_size_kb)
        if not scan.files:
            raise IndexingError(
                f"No supported files in {root} matching '{pattern}' "
                f"(supported: {', '.join(self.extensions)}; skipped {len(scan.skipped)})"
            )

        state = self.store.load_state()
        has_data = self.store.count() > 0
        reset = has_data and state.get("embedding_id") != self.embedding_id
        indexed = {} if reset else self.store.source_hashes()
        # Chunk size, overlap or splitting code changed: re-chunk every file (no reset
        # needed, the vectors stay comparable).
        rechunk = has_data and state.get("chunking_id") != self.chunker.config_id
        skip_if = {} if rechunk else indexed
        previous_empty = {} if reset or rechunk else state.get("empty_files", {})

        report = IndexReport(root=source_id(root), pattern=pattern, index_reset=reset)
        report.skipped_count = len(scan.skipped)
        report.skipped = [f"{p}: {why}" for p, why in scan.skipped[:MAX_REPORTED_SKIPS]]

        plan = IndexPlan(
            root=root,
            report=report,
            previous_roots=state.get("roots", []),
            previous_empty=previous_empty,
            rechunk=rechunk,
        )
        for file in scan.files:
            digest = hashlib.sha1(file.text.encode()).hexdigest()
            if digest in (skip_if.get(file.source), previous_empty.get(file.source)):
                report.files_unchanged += 1
                continue
            file_chunks = self.chunker.split(file)
            if not file_chunks:
                # Nothing to store in Chroma; remember the hash so that the file counts
                # as unchanged next time instead of being re-processed on every run.
                plan.empty[file.source] = digest
                if file.source in indexed:
                    plan.stale.append(file.source)  # drop its old chunks
                continue
            for chunk in file_chunks:
                chunk.metadata["file_hash"] = digest
                chunk.metadata["file_chunks"] = len(file_chunks)
            plan.changed.append(file.source)
            plan.chunks.extend(file_chunks)
            report.by_type[file.file_type] = report.by_type.get(file.file_type, 0) + 1
        if not reset:
            plan.stale.extend(self._stale_sources(root, indexed))
        return plan

    def apply(self, plan: IndexPlan, progress: ProgressCallback | None = None) -> IndexReport:
        """Embed and store the planned chunks. Slow: run it in the background."""
        started = time.perf_counter()
        report = plan.report
        if report.index_reset:
            logger.warning("Embedding model changed: rebuilding index as %s", self.embedding_id)
            self.store.reset()
        # Drop old chunks of changed files first: no duplicates, shortened files handled.
        self.store.delete_sources(plan.changed + plan.stale)
        report.files_removed = len(plan.stale)
        # Record the vector space before writing vectors: if this run is interrupted,
        # the next one must keep (not wipe) the chunks already written.
        early = {**self.store.load_state(), "embedding_id": self.embedding_id}
        if not plan.rechunk:
            # Interrupted ordinary run: files written so far are final and are skipped next
            # time. An interrupted re-chunk keeps the old chunking id so it is redone fully.
            early["chunking_id"] = self.chunker.config_id
        self.store.save_state(early)

        self._embed_and_store(plan.chunks, progress)
        report.files_indexed = len(plan.changed)
        report.chunks_written = len(plan.chunks)

        hashes = self.store.source_hashes()
        report.total_files = len(hashes)
        report.total_chunks = self.store.count()
        report.duration_s = round(time.perf_counter() - started, 2)

        previous_roots = [] if report.index_reset else plan.previous_roots
        rewritten = {*plan.changed, *plan.stale}
        empty_files = {
            source: digest
            for source, digest in plan.previous_empty.items()
            if source not in rewritten and _source_path(source).exists()
        } | plan.empty
        self.store.save_state(
            {
                "embedding_id": self.embedding_id,
                "chunking_id": self.chunker.config_id,
                "empty_files": empty_files,
                "last_indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "roots": sorted({*previous_roots, report.root}),
                "files_by_type": dict(Counter(s.rsplit(".", 1)[-1] for s in hashes)),
                "last_run": report.to_dict(),
            }
        )
        logger.info(
            "Indexed %s: %d changed, %d unchanged, %d removed, %d chunks in %.1fs",
            plan.root,
            report.files_indexed,
            report.files_unchanged,
            report.files_removed,
            report.chunks_written,
            report.duration_s,
        )
        return report

    def run(
        self, root: Path, pattern: str = "**/*", progress: ProgressCallback | None = None
    ) -> IndexReport:
        return self.apply(self.plan(root, pattern), progress)

    def _stale_sources(self, root: Path, indexed: dict[str, str]) -> list[str]:
        """Previously indexed files under ``root`` that no longer exist on disk."""
        root_id = source_id(root)
        prefix = "" if root_id == "." else root_id.rstrip("/") + "/"
        stale = []
        for source in indexed:
            if not source.startswith(prefix) or (not prefix and Path(source).is_absolute()):
                continue
            if not _source_path(source).exists():
                stale.append(source)
        return stale

    def _embed_and_store(self, chunks: list[Document], progress: ProgressCallback | None) -> None:
        total, done = len(chunks), 0
        for batch in batched(chunks, self.batch_size):
            vectors = self.embeddings.embed_documents([search_text(d) for d in batch])
            self.store.upsert(batch, vectors)
            done += len(batch)
            if progress:
                progress(done, total, f"Embedded {done}/{total} chunks")


def _source_path(source: str) -> Path:
    path = Path(source)
    return path if path.is_absolute() else Path.cwd() / path


def _check_pattern(pattern: str) -> None:
    """The glob must stay inside the indexed folder."""
    parts = Path(pattern).parts
    if Path(pattern).is_absolute() or pattern.startswith(("/", "\\")) or ".." in parts:
        raise IndexingError(
            f"Invalid glob '{pattern}': use a pattern relative to the folder without '..', "
            "e.g. '**/*' or 'docs/**/*.md'"
        )
