"""Shared fixtures. Tests never touch the network or real models."""

from pathlib import Path

import pytest

from helpers import SAMPLE_FILES, HashEmbeddings, write_files
from rag_kb.config import Settings
from rag_kb.knowledge_base import KnowledgeBase


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    return write_files(tmp_path / "docs", SAMPLE_FILES)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        chroma_path=tmp_path / "chroma",
        chroma_collection=f"test_{tmp_path.name[-20:]}".replace("-", "_"),
        chunk_size=300,
        chunk_overlap=40,
        embedding_batch_size=8,
    )


@pytest.fixture
def kb(settings: Settings) -> KnowledgeBase:
    return KnowledgeBase(settings, embeddings=HashEmbeddings())
