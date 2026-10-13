from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_kb.config import Settings


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults():
    s = make()
    assert s.embedding_model == "intfloat/multilingual-e5-large"
    assert s.llm_provider == "ollama"
    assert ".md" in s.supported_extensions


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://example:1234/v1")
    monkeypatch.setenv("MAX_RETRIES", "1")
    s = make()
    assert s.llm_provider == "openai"
    assert s.max_retries == 1
    assert s.effective_embedding_base_url == "http://example:1234/v1"


def test_embedding_url_explicit():
    s = make(embedding_base_url="http://emb:11434")
    assert s.effective_embedding_base_url == "http://emb:11434"


def test_env_example_matches_defaults():
    """.env.example must parse and document the real defaults."""
    from_example = Settings(_env_file=Path(__file__).parents[1] / ".env.example")
    assert from_example == make()


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 200, "chunk_overlap": 200},
        {"bm25_weight": 0, "vector_weight": 0},
        {"retrieve_k": 10, "max_top_k": 5},
        {"llm_provider": "anthropic"},
    ],
)
def test_invalid(overrides):
    with pytest.raises(ValidationError):
        make(**overrides)


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_chat_models_cap_reply_length(provider):
    from rag_kb.llm import create_chat_models

    models = create_chat_models(make(llm_provider=provider, llm_base_url="http://x:1/v1"))
    key = "num_predict" if provider == "ollama" else "max_tokens"
    assert getattr(models.chat, key) == 1024
    assert getattr(models.json, key) == 384
