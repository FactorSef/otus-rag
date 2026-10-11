"""Background indexing jobs.

Embedding a large folder on CPU takes minutes, longer than MCP clients wait for a tool
call. ``index_folder`` therefore validates and plans synchronously, then embeds in a
background thread; progress is exposed through ``index_status``.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from rag_kb.indexing.indexer import IndexPlan, IndexReport

logger = logging.getLogger(__name__)

JobState = Literal["running", "completed", "failed"]


class IndexingInProgressError(RuntimeError):
    """Another indexing job is still running."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class IndexJob:
    root: str
    pattern: str
    files_to_index: int
    chunks_total: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: JobState = "running"
    chunks_done: int = 0
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None
    report: IndexReport | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def progress(self, done: int, total: int, message: str) -> None:
        self.chunks_done = done

    def wait(self, timeout: float | None) -> bool:
        """Block until the job finishes; ``True`` if it did within ``timeout``."""
        return self._done.wait(timeout)

    @property
    def eta_s(self) -> float | None:
        if self.state != "running" or not self.chunks_done:
            return None
        elapsed = time.perf_counter() - self._t0
        rate = self.chunks_done / elapsed
        return round((self.chunks_total - self.chunks_done) / rate, 1)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.id,
            "state": self.state,
            "root": self.root,
            "pattern": self.pattern,
            "files_to_index": self.files_to_index,
            "chunks_total": self.chunks_total,
            "chunks_done": self.chunks_done,
            "started_at": self.started_at,
        }
        if self.state == "running":
            percent = 100 * self.chunks_done / self.chunks_total if self.chunks_total else 100
            data["percent"] = round(percent, 1)
            data["eta_s"] = self.eta_s
        else:
            data["finished_at"] = self.finished_at
        if self.error:
            data["error"] = self.error
        if self.report:
            data["report"] = self.report.to_dict()
        return data


class JobRunner:
    """Runs at most one indexing job at a time in a daemon thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: IndexJob | None = None

    @property
    def current(self) -> IndexJob | None:
        """The running job, or the last finished one."""
        return self._current

    @property
    def busy(self) -> bool:
        job = self._current
        return job is not None and job.state == "running"

    def submit(
        self,
        plan: IndexPlan,
        apply: Callable[[IndexPlan, Callable[[int, int, str], None]], IndexReport],
        on_finish: Callable[[], None],
    ) -> IndexJob:
        with self._lock:
            if self.busy:
                job = self._current
                assert job is not None
                raise IndexingInProgressError(
                    f"Indexing of '{job.root}' is already running "
                    f"({job.chunks_done}/{job.chunks_total} chunks). "
                    "Check progress with index_status and retry when it finishes."
                )
            job = IndexJob(
                root=plan.report.root,
                pattern=plan.report.pattern,
                files_to_index=len(plan.changed),
                chunks_total=len(plan.chunks),
            )
            self._current = job

        def work() -> None:
            state: JobState = "completed"
            try:
                job.report = apply(plan, job.progress)
            except Exception as exc:  # reported through index_status
                logger.exception("Indexing job %s failed", job.id)
                job.error = f"{type(exc).__name__}: {exc}"
                state = "failed"
            try:
                on_finish()  # runs after partial work too: what was written is searchable
            except Exception as exc:
                logger.exception("Post-indexing hook failed")
                job.error = job.error or f"{type(exc).__name__}: {exc}"
                state = "failed"
            # Published last, so "completed" means the new data is already searchable.
            job.finished_at = _now()
            job.state = state
            job._done.set()

        threading.Thread(target=work, name=f"index-{job.id}", daemon=True).start()
        return job
