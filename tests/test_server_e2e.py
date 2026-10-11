"""End-to-end: MCP client -> FastMCP server -> knowledge base + graph (mock LLM).

Follows the grader's checklist from the assignment: index_status (empty) ->
index_folder -> index_status (stats) -> ask_question -> find_relevant_docs.
"""

import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from helpers import HashEmbeddings, ScriptedChatModel
from rag_kb.graph import prompts
from rag_kb.knowledge_base import KnowledgeBase
from rag_kb.llm import ChatModels, LLMUnavailableError
from rag_kb.server import create_server

LLM_OK = {"reachable": True, "model_available": True}


def fake_llm() -> ScriptedChatModel:
    def handler(system: str, user: str) -> str:
        if system == prompts.REWRITE_SYSTEM:
            return '{"ru": "e-steps Сойка", "en": "Soyka e-steps"}'
        if system == prompts.GRADE_SYSTEM:
            return '{"relevant": "yes"}' if "97.3" in user else '{"relevant": "no"}'
        if system == prompts.BROADEN_SYSTEM:
            return '{"queries": ["экструдер Сойка"]}'
        return "У принтера Сойка e-steps = 97.3 [1]."

    return ScriptedChatModel(handler=handler)


@pytest.fixture
def server(settings):
    kb = KnowledgeBase(settings, embeddings=HashEmbeddings())
    llm = fake_llm()
    return create_server(
        settings,
        kb=kb,
        models=ChatModels(chat=llm, json=llm),
        llm_check=lambda: None,
        llm_status=lambda: LLM_OK,
    )


def data(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_tools_are_listed_with_descriptions(server):
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {"index_folder", "ask_question", "find_relevant_docs", "index_status"}
    for tool in tools.values():
        assert len(tool.description) > 150
        assert "3D printing" in tool.description  # topic tells the agent when to use it
    assert "path" in tools["index_folder"].input_schema["properties"]
    assert tools["find_relevant_docs"].input_schema["properties"]["top_k"]["maximum"] == 50


async def test_grader_checklist(server, docs_dir: Path):
    async with Client(server) as client:
        status = data(await client.call_tool("index_status"))
        assert status["empty"] is True and status["chunks"] == 0
        assert status["llm"] == LLM_OK

        indexed = data(await client.call_tool("index_folder", {"path": str(docs_dir)}))
        assert indexed["state"] == "completed"
        assert indexed["report"]["files_indexed"] == 7

        status = data(await client.call_tool("index_status"))
        assert status["files"] == 7 and status["chunks"] > 7
        assert status["indexing"]["state"] == "completed"

        progress: list[str] = []

        async def on_progress(value, total, message):
            progress.append(message)

        answer = data(
            await client.call_tool(
                "ask_question", {"question": "Какой e-steps у Сойки?"}, progress_handler=on_progress
            )
        )
        assert "97.3" in answer["answer"]
        assert answer["sources"]
        assert any(s["source"].endswith(("fleet.yaml", "notes.txt")) for s in answer["sources"])
        assert answer["steps"][0].startswith("rewrite")
        assert progress and progress[-1] == "Generated the answer"

        found = data(
            await client.call_tool("find_relevant_docs", {"query": "PURGE_LINE_LENGTH_MM"})
        )
        top = found["results"][0]
        assert top["source"].endswith("tools/purge.py")
        assert top["rank"] == 1 and top["bm25_rank"] == 1
        assert "PURGE_LINE_LENGTH_MM" in top["text"]


async def test_errors_are_reported_as_tool_errors(server, docs_dir: Path):
    async with Client(server) as client:
        with pytest.raises(ToolError, match="empty"):
            await client.call_tool("ask_question", {"question": "что угодно"})
        with pytest.raises(ToolError, match="empty"):
            await client.call_tool("find_relevant_docs", {"query": "что угодно"})
        with pytest.raises(ToolError, match="Folder not found"):
            await client.call_tool("index_folder", {"path": str(docs_dir / "missing")})
        with pytest.raises(ToolError):
            await client.call_tool("find_relevant_docs", {"query": "x", "top_k": 0})


async def test_llm_unavailable_message(settings, docs_dir: Path):
    def unavailable():
        raise LLMUnavailableError("LLM is not available at http://nowhere:11434")

    kb = KnowledgeBase(settings, embeddings=HashEmbeddings())
    kb.index_folder(docs_dir)
    server = create_server(
        settings, kb=kb, llm_check=unavailable, llm_status=lambda: {"reachable": False}
    )
    async with Client(server) as client:
        with pytest.raises(ToolError, match="nowhere:11434"):
            await client.call_tool("ask_question", {"question": "Какой e-steps у Сойки?"})
        # search does not need the LLM
        found = data(await client.call_tool("find_relevant_docs", {"query": "Сойка"}))
        assert found["results"]


async def test_health_endpoint(server):
    import httpx

    app = server.http_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "chunks": 0}


async def test_invalid_glob_is_a_tool_error(server, docs_dir: Path):
    async with Client(server) as client:
        with pytest.raises(ToolError, match="Invalid glob"):
            await client.call_tool("index_folder", {"path": str(docs_dir), "glob": "../*"})


def test_check_llm_never_raises_on_odd_replies(settings, monkeypatch):
    import httpx

    from rag_kb.llm import check_llm

    class Odd:
        def raise_for_status(self):
            pass

        def json(self):
            return ["not", "an", "object"]

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Odd())
    status = check_llm(settings)
    assert status["reachable"] is False
    assert "AttributeError" in status["error"]


class SlowEmbeddings(HashEmbeddings):
    """Lets the test decide when each embedding batch is computed."""

    def __init__(self) -> None:
        import threading

        self.gate = threading.Semaphore(0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert self.gate.acquire(timeout=10)
        return super().embed_documents(texts)


async def test_partial_index_results_carry_a_warning(settings, docs_dir: Path):
    emb = SlowEmbeddings()
    kb = KnowledgeBase(settings.model_copy(update={"embedding_batch_size": 2}), embeddings=emb)
    llm = fake_llm()
    server = create_server(
        settings,
        kb=kb,
        models=ChatModels(chat=llm, json=llm),
        llm_check=lambda: None,
        llm_status=lambda: LLM_OK,
    )
    job = kb.start_indexing(docs_dir)
    emb.gate.release(2)  # a few chunks are searchable, most are not yet
    while job.chunks_done < 4:
        assert not job.wait(0.01)

    async with Client(server) as client:
        found = data(await client.call_tool("find_relevant_docs", {"query": "Сойка"}))
        assert "partial index" in found["warning"]
        assert "4/" in found["warning"]
        answer = data(await client.call_tool("ask_question", {"question": "Какой e-steps?"}))
        assert "partial index" in answer["warning"]

        emb.gate.release(100)
        assert job.wait(10)
        found = data(await client.call_tool("find_relevant_docs", {"query": "Сойка"}))
        assert "warning" not in found
