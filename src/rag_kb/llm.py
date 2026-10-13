"""Chat model factory and availability check for the external LLM.

The graph only depends on LangChain's ``BaseChatModel``; the provider is chosen by
``LLM_PROVIDER``: ``ollama`` (native API) or ``openai`` (any OpenAI-compatible server:
OpenAI, LM Studio, vLLM, llama.cpp server...).
"""

import logging
from dataclasses import dataclass

import httpx
from langchain_core.language_models import BaseChatModel

from rag_kb.config import Settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The LLM endpoint cannot be reached or does not serve the configured model."""


@dataclass(frozen=True)
class ChatModels:
    chat: BaseChatModel  # free-form answers
    json: BaseChatModel  # structured answers (JSON mode where the provider supports it)


def create_chat_models(settings: Settings) -> ChatModels:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        common = {
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_num_ctx,
            "client_kwargs": {"timeout": settings.llm_timeout_s},
        }
        return ChatModels(
            chat=ChatOllama(**common, num_predict=settings.llm_max_tokens),
            json=ChatOllama(**common, format="json", num_predict=settings.llm_json_max_tokens),
        )

    from langchain_openai import ChatOpenAI

    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
    # JSON mode is not supported uniformly by OpenAI-compatible servers; prompts ask for
    # JSON and parsing is tolerant, so the same model serves both roles.
    common = {
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "api_key": key,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_s,
        "max_retries": 1,
    }
    return ChatModels(
        chat=ChatOpenAI(**common, max_tokens=settings.llm_max_tokens),
        json=ChatOpenAI(**common, max_tokens=settings.llm_json_max_tokens),
    )


def check_llm(settings: Settings, timeout: float = 5.0) -> dict:
    """Probe the endpoint and verify the model is available. Never raises."""
    base = settings.llm_base_url.rstrip("/")
    info = {"provider": settings.llm_provider, "base_url": base, "model": settings.llm_model}
    try:
        if settings.llm_provider == "ollama":
            resp = httpx.get(f"{base}/api/tags", timeout=timeout)
            resp.raise_for_status()
            names = {m["name"] for m in resp.json().get("models", [])}
            available = _model_listed(settings.llm_model, names)
            hint = f"run: ollama pull {settings.llm_model}"
        else:
            headers = {}
            if settings.llm_api_key:
                headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
            resp = httpx.get(f"{base}/models", headers=headers, timeout=timeout)
            resp.raise_for_status()
            names = {m["id"] for m in resp.json().get("data", [])}
            available = _model_listed(settings.llm_model, names)
            hint = "load the model on the server or fix LLM_MODEL"
    except Exception as exc:  # any malformed reply means "not usable", never a crash
        return {**info, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}

    info.update(reachable=True, model_available=available)
    if not available:
        info["error"] = f"model '{settings.llm_model}' not found ({hint})"
        info["available_models"] = sorted(names)[:20]
    return info


def ensure_llm(settings: Settings) -> None:
    status = check_llm(settings)
    if not status.get("reachable") or not status.get("model_available"):
        raise LLMUnavailableError(
            f"LLM is not available at {status['base_url']} "
            f"(provider {status['provider']}, model {status['model']}): {status.get('error')}. "
            "Set LLM_BASE_URL / LLM_MODEL or start Ollama "
            "(docker compose --profile local-llm up)."
        )


def _model_listed(model: str, names: set[str]) -> bool:
    # Ollama lists "qwen2.5:3b"; a bare name means the ":latest" tag.
    candidates = {model, f"{model}:latest"} if ":" not in model else {model}
    return bool(candidates & names)
