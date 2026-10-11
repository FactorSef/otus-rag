# RAG Knowledge Base — MCP‑сервер

MCP‑сервер, превращающий папку с документами в поисковую базу знаний: Corrective RAG на LangGraph, гибридный поиск (BM25 + вектор → RRF), внешняя LLM (Ollama или OpenAI‑совместимый API).

> Документ в работе. Архитектура — [ARCHITECTURE.md](ARCHITECTURE.md).

## Разработка

```bash
uv sync
cp .env.example .env
uv run pytest
uv run ruff check .
```
