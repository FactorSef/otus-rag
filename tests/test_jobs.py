"""Background indexing: progress, concurrency, failure and recovery."""

import threading
from pathlib import Path

import pytest

from helpers import HashEmbeddings
from rag_kb.indexing.jobs import IndexingInProgressError
from rag_kb.knowledge_base import KnowledgeBase

TIMEOUT = 10


class GatedEmbeddings(HashEmbeddings):
    """Embeds one batch per ``release()``; optionally fails on the N-th batch."""

    def __init__(self, fail_on_batch: int | None = None) -> None:
        self.gate = threading.Semaphore(0)
        self.fail_on_batch = fail_on_batch
        self.batches = 0
        self.gated = True

    def release(self, n: int = 1) -> None:
        for _ in range(n):
            self.gate.release()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.gated:
            assert self.gate.acquire(timeout=TIMEOUT), "test never released the batch"
        self.batches += 1
        if self.batches == self.fail_on_batch:
            raise RuntimeError("embedding backend crashed")
        return super().embed_documents(texts)


@pytest.fixture
def small_batches(settings):
    return settings.model_copy(update={"embedding_batch_size": 2})


def test_job_runs_in_background_and_reports_progress(small_batches, docs_dir: Path):
    emb = GatedEmbeddings()
    kb = KnowledgeBase(small_batches, embeddings=emb)

    job = kb.start_indexing(docs_dir)  # returns before any embedding happened
    assert job.state == "running"
    assert job.files_to_index == 7
    batches = -(-job.chunks_total // 2)
    assert kb.status()["indexing"]["chunks_done"] == 0

    emb.release(2)
    while job.chunks_done < 4:
        assert not job.wait(0.01)
    running = kb.status()["indexing"]
    assert running["state"] == "running"
    assert running["percent"] > 0
    assert running["eta_s"] is not None

    emb.release(batches)
    assert job.wait(TIMEOUT)
    done = kb.status()
    assert done["indexing"]["state"] == "completed"
    assert done["indexing"]["report"]["files_indexed"] == 7
    assert done["files"] == 7
    # BM25 was rebuilt when the job finished
    assert kb.search("PURGE_LINE_LENGTH_MM", 1)[0].bm25_rank == 1


def test_second_job_rejected_while_running(small_batches, docs_dir: Path):
    emb = GatedEmbeddings()
    kb = KnowledgeBase(small_batches, embeddings=emb)
    job = kb.start_indexing(docs_dir)
    with pytest.raises(IndexingInProgressError):
        kb.start_indexing(docs_dir)
    emb.release(100)
    assert job.wait(TIMEOUT)
    assert job.state == "completed"


def test_failed_job_is_reported_and_next_run_recovers(small_batches, docs_dir: Path):
    emb = GatedEmbeddings(fail_on_batch=3)
    emb.gated = False
    kb = KnowledgeBase(small_batches, embeddings=emb)

    job = kb.start_indexing(docs_dir)
    assert job.wait(TIMEOUT)
    assert job.state == "failed"
    assert "embedding backend crashed" in kb.status()["indexing"]["error"]
    partial_files = kb.status()["files"]
    assert partial_files < 7  # partly written files are not counted as indexed

    emb.fail_on_batch = None
    report = kb.index_folder(docs_dir)
    assert report.files_unchanged == partial_files  # completed files are kept
    assert report.files_indexed == 7 - partial_files
    assert kb.status()["files"] == 7
    assert kb.store.count() == report.total_chunks
    ids = [d.id for d in kb.store.all_documents()]
    assert len(ids) == len(set(ids))
