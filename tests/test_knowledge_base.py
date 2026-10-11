"""Indexer + hybrid search end to end (fake embeddings, real ChromaDB)."""

from pathlib import Path

import pytest

from helpers import HashEmbeddings
from rag_kb.indexing.indexer import IndexingError
from rag_kb.knowledge_base import EmptyIndexError, KnowledgeBase


def test_empty_status_and_search(kb: KnowledgeBase):
    status = kb.status()
    assert status["empty"] is True
    assert status["files"] == status["chunks"] == 0
    assert status["last_indexed_at"] is None
    with pytest.raises(EmptyIndexError):
        kb.search("anything", 5)


def test_index_folder_all_formats(kb: KnowledgeBase, docs_dir: Path):
    report = kb.index_folder(docs_dir)
    assert report.files_indexed == 7
    assert set(report.by_type) == {"md", "txt", "py", "js", "ts", "json", "yaml"}
    assert report.total_chunks == kb.store.count() > 7

    status = kb.status()
    assert status["files"] == 7
    assert status["chunks"] == report.total_chunks
    assert status["last_indexed_at"]
    assert status["files_by_type"]["py"] == 1


def test_reindex_is_idempotent_and_incremental(kb: KnowledgeBase, docs_dir: Path):
    first = kb.index_folder(docs_dir)
    second = kb.index_folder(docs_dir)
    assert second.files_indexed == 0
    assert second.files_unchanged == 7
    assert kb.store.count() == first.total_chunks  # no duplicates

    (docs_dir / "notes.txt").write_text("Совсем другой текст.", encoding="utf-8")
    (docs_dir / "tools" / "temp.js").unlink()
    third = kb.index_folder(docs_dir)
    assert third.files_indexed == 1
    assert third.files_removed == 1
    assert third.total_files == 6
    sources = {d.metadata["source"] for d in kb.store.all_documents()}
    assert not any(s.endswith("temp.js") for s in sources)


def test_glob_pattern_limits_files(kb: KnowledgeBase, docs_dir: Path):
    report = kb.index_folder(docs_dir, "tools/*")
    assert report.files_indexed == 3
    assert set(report.by_type) == {"py", "js", "ts"}


@pytest.mark.parametrize("bad", ["does/not/exist", "file"])
def test_invalid_paths(kb: KnowledgeBase, docs_dir: Path, bad: str):
    path = docs_dir / "guide.md" if bad == "file" else docs_dir / bad
    with pytest.raises(IndexingError):
        kb.index_folder(path)


def test_no_supported_files(kb: KnowledgeBase, tmp_path: Path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"\x89PNG")
    with pytest.raises(IndexingError, match="No supported files"):
        kb.index_folder(folder)


def test_hybrid_search_finds_identifier_and_reports_ranks(kb: KnowledgeBase, docs_dir: Path):
    kb.index_folder(docs_dir)
    hits = kb.search("PURGE_LINE_LENGTH_MM", 3)
    top = hits[0]
    assert top.doc.metadata["source"].endswith("tools/purge.py")
    assert top.bm25_rank == 1
    assert top.vector_rank is not None
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_hybrid_search_russian(kb: KnowledgeBase, docs_dir: Path):
    kb.index_folder(docs_dir)
    hits = kb.search("какой e-steps у принтера Сойка?", 3)
    assert {h.doc.metadata["source"].rsplit("/", 1)[-1] for h in hits[:2]} & {
        "notes.txt",
        "fleet.yaml",
    }


def test_bm25_rebuilt_after_restart(settings, docs_dir: Path):
    KnowledgeBase(settings, embeddings=HashEmbeddings()).index_folder(docs_dir)
    restarted = KnowledgeBase(settings, embeddings=HashEmbeddings())
    assert restarted.status()["files"] == 7
    assert restarted.search("Smoky Quartz PETG", 1)[0].bm25_rank == 1


def test_embedding_model_change_rebuilds_index(settings, docs_dir: Path):
    KnowledgeBase(settings, embeddings=HashEmbeddings()).index_folder(docs_dir)
    changed = settings.model_copy(update={"embedding_model": "other-model"})
    kb2 = KnowledgeBase(changed, embeddings=HashEmbeddings())
    assert "warning" in kb2.status()
    report = kb2.index_folder(docs_dir)
    assert report.index_reset is True
    assert report.files_indexed == 7
    assert "warning" not in kb2.status()


@pytest.mark.parametrize("pattern", ["../**/*.md", "/etc/*", "sub/../../*"])
def test_glob_must_stay_inside_folder(kb: KnowledgeBase, docs_dir: Path, pattern: str):
    with pytest.raises(IndexingError, match="Invalid glob"):
        kb.index_folder(docs_dir, pattern)


def test_chunk_settings_change_rechunks_unchanged_files(settings, docs_dir: Path):
    KnowledgeBase(settings, embeddings=HashEmbeddings()).index_folder(docs_dir)
    smaller = settings.model_copy(update={"chunk_size": 200, "chunk_overlap": 20})
    kb2 = KnowledgeBase(smaller, embeddings=HashEmbeddings())
    report = kb2.index_folder(docs_dir)
    assert report.files_indexed == 7  # content unchanged, chunking changed
    assert all(len(d.page_content) <= 200 for d in kb2.store.all_documents())
    assert kb2.index_folder(docs_dir).files_indexed == 0  # and then it settles


def test_file_without_chunks_is_not_reprocessed(kb: KnowledgeBase, docs_dir: Path):
    (docs_dir / "empty.json").write_text("{}", encoding="utf-8")
    kb.index_folder(docs_dir)
    again = kb.index_folder(docs_dir)
    assert again.files_indexed == 0
    assert again.files_unchanged == 8


def test_search_refuses_index_of_another_model(settings, docs_dir: Path):
    KnowledgeBase(settings, embeddings=HashEmbeddings()).index_folder(docs_dir)
    other = settings.model_copy(update={"embedding_model": "other-model"})
    with pytest.raises(EmptyIndexError, match="rebuild"):
        KnowledgeBase(other, embeddings=HashEmbeddings()).search("Сойка", 3)


def test_bm25_follows_collection_reset(settings, docs_dir: Path):
    kb = KnowledgeBase(settings, embeddings=HashEmbeddings())
    kb.index_folder(docs_dir)
    kb.store.reset()  # what a running rebuild does when the embedding model changes
    kb.store.upsert([kb._bm25.docs[0]], [HashEmbeddings().embed_query("x")])
    kb.store.save_state({"embedding_id": kb.embedding_id})
    sources = {h.doc.id for h in kb.search("PURGE_LINE_LENGTH_MM Сойка", 10)}
    assert sources == {kb._bm25.docs[0].id}  # no hits from wiped chunks
