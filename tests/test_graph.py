"""Corrective RAG graph with a mock LLM and a fake retriever."""

import json

import pytest
from langchain_core.documents import Document

from helpers import ScriptedChatModel
from rag_kb.config import Settings
from rag_kb.graph import prompts
from rag_kb.graph.builder import build_graph
from rag_kb.graph.nodes import parse_relevance, question_language

QUESTION = "Какой e-steps у принтера Сойка?"


def chunk(i: int, text: str) -> Document:
    return Document(
        id=f"c{i}",
        page_content=text,
        metadata={"source": f"lab/doc{i}.md", "start_line": i, "end_line": i + 2, "section": ""},
    )


RELEVANT = [chunk(1, "Сойка: e_steps 97.3, сопло 0.4"), chunk(2, "Soyka extruder e-steps = 97.3")]
NOISE = [chunk(3, "Сушка филамента PETG"), chunk(4, "Klipper pressure advance")]
HIDDEN = [chunk(5, "Калибровка экструдера Сойки: 97.3 шага/мм"), chunk(6, "Soyka calibration log")]


class FakeSearch:
    """Maps query keywords to ranked chunks and records the queries."""

    def __init__(self, table: dict[str, list[Document]], default: list[Document]) -> None:
        self.table, self.default, self.queries = table, default, []

    def __call__(self, query: str, k: int) -> list[Document]:
        self.queries.append(query)
        for key, docs in self.table.items():
            if key in query.lower():
                return docs[:k]
        return self.default[:k]


def llm_for(
    relevant_ids: set[str],
    rewrite='{"ru": "Сойка e-steps", "en": "Soyka e-steps"}',
    broaden='{"queries": ["калибровка экструдера", "extruder calibration"]}',
):
    def handler(system: str, user: str) -> str:
        if system == prompts.REWRITE_SYSTEM:
            return rewrite
        if system == prompts.BROADEN_SYSTEM:
            return broaden
        if system == prompts.GRADE_SYSTEM:
            ok = any(f"doc{i[1:]}.md" in user for i in relevant_ids)
            return json.dumps({"relevant": "yes" if ok else "no"})
        return "У Сойки e-steps равен 97.3 [1]."

    return ScriptedChatModel(handler=handler)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, retrieve_k=4, min_relevant=2, max_retries=2)


def run(llm, search, settings):
    return build_graph(llm, search, settings).invoke({"question": QUESTION})


def test_direct_path_without_retries(settings):
    llm = llm_for({"c1", "c2"})
    search = FakeSearch({}, default=[*RELEVANT, *NOISE])
    result = run(llm, search, settings)

    steps = [s.split(":")[0] for s in result["steps"]]
    assert steps == ["rewrite", "retrieve", "grade", "generate"]
    assert result["retries"] == 0
    assert "97.3" in result["answer"]
    assert [s["source"] for s in result["sources"]] == ["lab/doc1.md", "lab/doc2.md"]
    assert result["sources"][0]["lines"] == "1-3"
    # original question is searched together with the rewritten RU/EN queries
    assert search.queries == [QUESTION, "Сойка e-steps", "Soyka e-steps"]
    assert len(llm.calls_to(prompts.GRADE_SYSTEM)) == 4  # every chunk graded


def test_broaden_loop_finds_more(settings):
    llm = llm_for({"c1", "c5", "c6"})
    search = FakeSearch(
        {"калибровка": HIDDEN, "calibration": HIDDEN}, default=[RELEVANT[0], *NOISE]
    )
    result = run(llm, search, settings)

    steps = [s.split(":")[0] for s in result["steps"]]
    assert steps == ["rewrite", "retrieve", "grade", "broaden #1", "retrieve", "grade", "generate"]
    assert result["retries"] == 1
    assert {d.id for d in result["relevant"]} == {"c1", "c5", "c6"}
    # chunks graded in the first loop are not graded again
    graded = llm.calls_to(prompts.GRADE_SYSTEM)
    assert len(graded) == len(set(graded)) == 5


def test_retry_limit_then_not_found(settings):
    llm = llm_for(set())
    search = FakeSearch({}, default=NOISE)
    result = run(llm, search, settings)

    assert result["retries"] == settings.max_retries
    assert sum(s.startswith("broaden") for s in result["steps"]) == 2
    assert result["answer"] == prompts.NOT_FOUND["Russian"]
    assert result["sources"] == []
    assert not llm.calls_to(prompts.GENERATE_SYSTEM[:20])  # no answer generation call


def test_partial_relevance_after_budget_still_answers(settings):
    llm = llm_for({"c1"})
    search = FakeSearch({}, default=[RELEVANT[0], *NOISE])
    result = run(llm, search, settings.model_copy(update={"max_retries": 1}))
    assert result["retries"] == 1
    assert len(result["sources"]) == 1  # one relevant chunk < MIN_RELEVANT, answered anyway
    assert "97.3" in result["answer"]


def test_garbage_llm_output_does_not_break_graph(settings):
    def handler(system: str, user: str) -> str:
        if system == prompts.GRADE_SYSTEM:
            return "Yes, this fragment is relevant." if "doc1" in user else "blah"
        if system.startswith("You answer"):
            return "Ответ [1]"
        return "not json at all"

    llm = ScriptedChatModel(handler=handler)
    search = FakeSearch({}, default=[RELEVANT[0], *NOISE])
    result = run(llm, search, settings)

    assert search.queries[0] == QUESTION  # rewrite fell back to the question itself
    assert result["retries"] == 2  # broaden produced nothing usable but the loop continued
    assert [s["source"] for s in result["sources"]] == ["lab/doc1.md"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"relevant": "yes"}', True),
        ('```json\n{"relevant": "no"}\n```', False),
        ('{"relevant": true}', True),
        ("Yes.", True),
        ("Нет, не относится", False),
        ("", False),
        ("I am not sure", False),
    ],
)
def test_parse_relevance(reply, expected):
    assert parse_relevance(reply) is expected


def test_question_language():
    assert question_language("Какой e-steps у Сойки?") == "Russian"
    assert question_language("What is PURGE_LINE_LENGTH_MM?") == "English"


def test_context_budget(settings):
    long_docs = [chunk(i, "x" * 400) for i in range(10)]
    llm = llm_for({d.id for d in long_docs})
    search = FakeSearch({}, default=long_docs)
    small = settings.model_copy(update={"max_context_chars": 1000, "retrieve_k": 10})
    result = run(llm, search, small)
    assert len(result["sources"]) == 2  # 2 x 400 fits, the 3rd would exceed 1000


def test_thinking_blocks_and_prompt_suffix(settings):
    def handler(system: str, user: str) -> str:
        assert user.endswith("\n/no_think")
        if system == prompts.GRADE_SYSTEM:
            return '<think>{"relevant": "no"} hmm, actually it fits</think>{"relevant": "yes"}'
        if system.startswith("You answer"):
            return "<think>reasoning</think>Ответ: 97.3 [1]"
        return '<think>...</think>{"ru": "Сойка", "en": "Soyka"}'

    llm = ScriptedChatModel(handler=handler)
    search = FakeSearch({}, default=RELEVANT)
    result = run(llm, search, settings.model_copy(update={"llm_prompt_suffix": "/no_think"}))
    assert result["answer"] == "Ответ: 97.3 [1]"
    assert len(result["relevant"]) == 2
    assert search.queries == [QUESTION, "Сойка", "Soyka"]
