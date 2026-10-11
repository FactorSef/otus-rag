"""MCP server exposing the knowledge base as four tools."""

import asyncio
import logging
from collections.abc import Callable
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from rag_kb.config import Settings, get_settings
from rag_kb.graph.builder import build_graph
from rag_kb.indexing.indexer import IndexingError
from rag_kb.indexing.jobs import IndexingInProgressError
from rag_kb.knowledge_base import EmptyIndexError, KnowledgeBase
from rag_kb.llm import ChatModels, LLMUnavailableError, check_llm, create_chat_models, ensure_llm

logger = logging.getLogger(__name__)

# Progress shown while ask_question runs: (node -> human readable step).
STEP_LABELS = {
    "rewrite": "Rewrote the question into search queries",
    "retrieve": "Searched the knowledge base (BM25 + vector)",
    "grade": "Graded retrieved chunks for relevance",
    "broaden": "Too few relevant chunks: broadening the query",
    "generate": "Generated the answer",
}


def _instructions(topic: str) -> str:
    return (
        f"Internal knowledge base (RAG) about {topic}. Documents are in Russian and English. "
        "Use these tools whenever a question may be answered by this knowledge base, BEFORE "
        "relying on general knowledge: it contains project-specific facts (exact numbers, "
        "names, settings, dates) that general knowledge does not have. Typical flow: "
        "index_status -> (index_folder if empty) -> ask_question or find_relevant_docs."
    )


def create_server(
    settings: Settings,
    kb: KnowledgeBase | None = None,
    models: ChatModels | None = None,
    llm_check: Callable[[], None] | None = None,
    llm_status: Callable[[], dict] | None = None,
) -> FastMCP:
    """Build the server. Dependencies are injectable for tests."""
    kb = kb or KnowledgeBase(settings)
    llm_check = llm_check or (lambda: ensure_llm(settings))
    llm_status = llm_status or (lambda: check_llm(settings, timeout=2.0))
    lazy_models: dict[str, ChatModels] = {}

    def get_graph():
        if "m" not in lazy_models:
            lazy_models["m"] = models or create_chat_models(settings)
        m = lazy_models["m"]

        def search(query: str, k: int):
            return [hit.doc for hit in kb.search(query, k)]

        return build_graph(m.chat, search, settings, json_llm=m.json)

    topic = settings.kb_topic
    mcp = FastMCP("rag-knowledge-base", instructions=_instructions(topic))

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> JSONResponse:
        """Liveness probe for Docker (does not touch the LLM)."""
        return JSONResponse({"status": "ok", "chunks": kb.store.count()})

    @mcp.tool(
        description=(
            f"Index a folder of documents into the knowledge base about {topic}. "
            "Scans files matching a glob pattern (.md, .txt, .py, .js, .ts, .json, .yaml), "
            "splits them into chunks, embeds them and stores them for hybrid search. "
            "Re-indexing is incremental: unchanged files are skipped, deleted files are "
            "removed. Large folders are indexed in the background: if the result has "
            "state='running', poll index_status until it is 'completed'. "
            "Paths are resolved on the server; the bundled demo documents are at "
            "'./sample_docs'."
        ),
        annotations={"title": "Index folder", "destructiveHint": False, "idempotentHint": True},
    )
    async def index_folder(
        path: Annotated[
            str, Field(description="Folder to index, e.g. './sample_docs' (server-side path)")
        ],
        glob: Annotated[
            str,
            Field(description="Glob pattern relative to the folder, e.g. '**/*' or '**/*.md'"),
        ] = "**/*",
    ) -> dict[str, Any]:
        try:
            job = kb.start_indexing(path, glob)
        except (IndexingError, IndexingInProgressError) as exc:
            raise ToolError(str(exc)) from exc
        finished = await asyncio.to_thread(job.wait, settings.index_wait_s)
        result = job.to_dict()
        if not finished:
            result["message"] = (
                f"Indexing {job.chunks_total} chunks from {job.files_to_index} files in the "
                "background. Call index_status to follow progress; searches use the "
                "already indexed data until it completes."
            )
        elif job.state == "failed":
            raise ToolError(f"Indexing failed: {job.error}")
        return result

    @mcp.tool(
        description=(
            f"Answer a question using the knowledge base about {topic}. "
            "Use this FIRST for any question on these topics, including questions about "
            "specific printers, settings, numbers, people, dates or internal procedures — "
            "the knowledge base contains facts that are not public. Runs a Corrective RAG "
            "pipeline (query rewrite -> hybrid search -> relevance grading -> query "
            "broadening if needed -> answer) with a local LLM and returns the answer with "
            "cited sources (file and line range). Works for Russian and English questions. "
            "Can take up to a minute or two on CPU."
        ),
        annotations={"title": "Ask the knowledge base", "readOnlyHint": True},
    )
    async def ask_question(
        question: Annotated[
            str, Field(description="Question in natural language (Russian or English)")
        ],
        ctx: Context,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ToolError("question must not be empty")
        if kb.is_empty:
            raise ToolError(
                "The knowledge base is empty"
                + (" (indexing in progress, see index_status)." if kb.jobs.busy else ": ")
                + ("" if kb.jobs.busy else "call index_folder('./sample_docs') first.")
            )
        try:
            await asyncio.to_thread(llm_check)
            graph = get_graph()
            state: dict[str, Any] = {}
            steps: list[str] = []
            async for update in graph.astream({"question": question}, stream_mode="updates"):
                for node, delta in update.items():
                    delta = dict(delta or {})
                    steps.extend(delta.pop("steps", []))  # appended, not replaced
                    state.update(delta)
                    # Progress notifications also keep clients from timing out.
                    await ctx.report_progress(len(steps), message=STEP_LABELS.get(node, node))
        except LLMUnavailableError as exc:
            raise ToolError(str(exc)) from exc
        except EmptyIndexError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            logger.exception("ask_question failed")
            raise ToolError(
                f"Failed to answer: {type(exc).__name__}: {exc}. "
                f"LLM: {settings.llm_provider} {settings.llm_model} at {settings.llm_base_url}"
            ) from exc

        return {
            "answer": state.get("answer", ""),
            "sources": state.get("sources", []),
            "relevant_chunks": len(state.get("relevant", [])),
            "retries": state.get("retries", 0),
            "steps": steps,
        }

    @mcp.tool(
        description=(
            f"Search the knowledge base about {topic} and return the most relevant document "
            "chunks WITHOUT generating an answer (no LLM needed, fast). Hybrid search: BM25 "
            "keyword matching (good for exact names, numbers, code identifiers like "
            "TOKEN_EXPIRY_HOURS) fused with multilingual semantic vector search via "
            "Reciprocal Rank Fusion. Use it to look up where something is mentioned, to "
            "quote sources, or when you want to read raw text yourself."
        ),
        annotations={"title": "Find relevant documents", "readOnlyHint": True},
    )
    async def find_relevant_docs(
        query: Annotated[str, Field(description="Search query: keywords or a question")],
        top_k: Annotated[
            int, Field(description="How many chunks to return", ge=1, le=settings.max_top_k)
        ] = 5,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ToolError("query must not be empty")
        try:
            hits = await asyncio.to_thread(kb.search, query, top_k)
        except EmptyIndexError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "query": query,
            "results": [
                {
                    "rank": rank,
                    "source": h.doc.metadata.get("source"),
                    "lines": f"{h.doc.metadata.get('start_line')}-{h.doc.metadata.get('end_line')}",
                    "section": h.doc.metadata.get("section", ""),
                    "score": round(h.score, 5),
                    "bm25_rank": h.bm25_rank,
                    "vector_rank": h.vector_rank,
                    "text": h.doc.page_content,
                }
                for rank, h in enumerate(hits, 1)
            ],
        }

    @mcp.tool(
        description=(
            f"Show the state of the knowledge base about {topic}: number of indexed files "
            "and chunks, files by type, time of the last indexing, progress of a running "
            "indexing job, embedding model and whether the LLM endpoint is reachable. "
            "Call it to check whether the knowledge base is ready before asking questions."
        ),
        annotations={"title": "Index status", "readOnlyHint": True},
    )
    async def index_status() -> dict[str, Any]:
        status = await asyncio.to_thread(kb.status)
        status["llm"] = await asyncio.to_thread(llm_status)
        return status

    return mcp


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    llm = check_llm(settings)
    if llm.get("reachable") and llm.get("model_available"):
        logger.info("LLM ready: %s %s at %s", llm["provider"], llm["model"], llm["base_url"])
    else:
        logger.warning("LLM not ready (ask_question will fail until it is): %s", llm)

    mcp = create_server(settings)
    if settings.server_transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=settings.server_host, port=settings.server_port)


if __name__ == "__main__":
    main()
