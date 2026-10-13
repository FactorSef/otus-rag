# CLAUDE.md

MCP server: Corrective RAG (LangGraph) over a local folder, hybrid BM25 + vector search.
Architecture — ARCHITECTURE.md, user docs — README.md (Russian).

## Commands

```bash
uv run pytest -q                                  # no network, no models (mock LLM)
uv run ruff check . && uv run ruff format --check .
uv run python eval/run_eval.py --index ./sample_docs   # quality eval, needs `docker compose up`
```

## Rules

- **After changing a prompt (`src/rag_kb/graph/prompts.py`), the graph (`src/rag_kb/graph/`)
  or the model / provider (LLM_* settings, `src/rag_kb/llm.py`) — run `eval/run_eval.py` on
  the default model and report its three numbers (retrieval, grader, answer).** Unit tests
  check graph routing with a mock LLM that knows the right answer; they say nothing about
  the quality of the real model's decisions. Record the numbers in REPORT.md
  («Замер качества»). If the eval cannot be run (no LLM / Docker), say so explicitly
  instead of claiming the change works.
- A result measured on one model is not valid for another: a model, provider or call-path
  change (JSON mode, `/no_think`) is a system change and needs a new eval run.
