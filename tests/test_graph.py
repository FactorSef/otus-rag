"""Corrective RAG graph with a mock LLM and a fake retriever."""

import json

import pytest
from langchain_core.documents import Document

from helpers import ScriptedChatModel
from rag_kb.config import Settings
from rag_kb.graph import prompts
from rag_kb.graph.builder import build_graph
from rag_kb.graph.nodes import (
    check_citations,
    parse_grade,
    parse_relevance,
    question_language,
    verified_quote,
)

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


def test_broaden_remembers_all_tried_queries_and_digs_deeper(settings):
    # Broaden always proposes the original rewrite queries: nothing new to try.
    llm = llm_for(set(), broaden='{"queries": ["Сойка e-steps", "Soyka e-steps"]}')
    ranked = [*RELEVANT, *NOISE, *HIDDEN]
    calls = []

    def search(query, k):
        calls.append(k)
        return ranked[:k]

    result = run(llm, search, settings)
    # Second broaden saw the rewrite queries as tried and did not repeat them.
    broadens = [s for s in result["steps"] if s.startswith("broaden")]
    assert all("Soyka e-steps" not in s for s in broadens)
    # Each retry searched deeper than the chunks already graded, so it found new ones.
    assert max(calls) > settings.retrieve_k
    graded = [s for s in result["steps"] if s.startswith("grade")]
    assert graded[1] != "grade: 0/0 relevant (total 0)"


# --- grader input/output, fallback, citation checks --------------------------------------


def ranked(doc: Document, bm25: int | None, vector: int | None) -> Document:
    return Document(
        id=doc.id,
        page_content=doc.page_content,
        metadata={**doc.metadata, "bm25_rank": bm25, "vector_rank": vector},
    )


def test_grader_sees_rewritten_queries(settings):
    llm = llm_for({"c1", "c2"})
    run(llm, FakeSearch({}, default=[*RELEVANT, *NOISE]), settings)
    for user in llm.calls_to(prompts.GRADE_SYSTEM):
        assert user.startswith(
            f"Question: {QUESTION}\nSearch queries: Сойка e-steps; Soyka e-steps"
        )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"quote": "e_steps_per_mm: 97.3", "relevant": "yes"}', (True, "e_steps_per_mm: 97.3")),
        ('{"quote": "none", "relevant": "no"}', (False, "none")),
        # unescaped quotes break the JSON: the verdict field still decides, not words in the quote
        ('{"quote": "say "yes" here", "relevant": "no"}', (False, 'say "yes" here')),
        ('{"quote": "x "no" y", "relevant": "yes"}', (True, 'x "no" y')),
        ('{"quote": "yes it is', (False, "")),
    ],
)
def test_parse_grade(reply, expected):
    assert parse_grade(reply) == expected


def test_verified_quote_rejects_invented_text():
    text = "extruder:\n  e_steps_per_mm: 97.3   # calibrated"
    assert verified_quote("e_steps_per_mm:  97.3", text) == "e_steps_per_mm: 97.3"
    assert verified_quote("e-steps равен 97.3", text) == ""
    assert verified_quote("none", text) == ""


def test_relevant_chunk_keeps_verified_quote_for_failed_generation(settings):
    def handler(system: str, user: str) -> str:
        if system == prompts.GRADE_SYSTEM:
            if "doc1.md" in user:
                return '{"quote": "e_steps 97.3", "relevant": "yes"}'
            if "doc2.md" in user:
                return '{"quote": "invented 12.5", "relevant": "yes"}'
            return '{"quote": "none", "relevant": "no"}'
        if system.startswith("You answer"):
            return "[2]"  # nothing but a citation
        return '{"ru": "Сойка", "en": "Soyka"}'

    result = run(ScriptedChatModel(handler=handler), FakeSearch({}, RELEVANT), settings)
    assert result["answer"].splitlines() == [
        prompts.ANSWER_FAILED["Russian"],
        "- «e_steps 97.3» [1]",
    ]
    assert "generate: answer had no content besides citations" in result["steps"]


def test_citations_of_missing_fragments_are_dropped():
    answer, dropped, has_content = check_citations("E-steps 97.3 [1][4]. Сопло 0.4 [7].", 2)
    assert (answer, dropped, has_content) == ("E-steps 97.3 [1]. Сопло 0.4.", [4, 7], True)
    assert check_citations("[2]", 2) == ("[2]", [], False)
    assert check_citations("[3]", 2) == ("", [3], False)


def test_fallback_uses_top_ranked_chunks_rejected_by_grader(settings):
    llm = llm_for(set())  # the grader rejects everything
    strong = ranked(RELEVANT[0], bm25=2, vector=3)
    weak = [ranked(NOISE[0], bm25=1, vector=None), ranked(NOISE[1], bm25=15, vector=1)]
    result = run(llm, FakeSearch({}, default=[strong, *weak]), settings)

    assert result["low_confidence"] is True
    assert [s["source"] for s in result["sources"]] == ["lab/doc1.md"]
    assert result["answer"].startswith(prompts.LOW_CONFIDENCE_NOTE["Russian"])
    assert any("low confidence" in s for s in result["steps"])


def test_fallback_can_be_disabled(settings):
    search = FakeSearch({}, default=[ranked(RELEVANT[0], bm25=1, vector=1)])
    result = run(llm_for(set()), search, settings.model_copy(update={"fallback_max_rank": 0}))
    assert result["answer"] == prompts.NOT_FOUND["Russian"]
    assert result["low_confidence"] is False


def test_retrieve_keeps_best_rank_over_queries(settings):
    # The same chunk is found by two queries with different per-list ranks.
    table = {
        "сойка": [ranked(RELEVANT[0], bm25=8, vector=2)],
        "soyka": [ranked(RELEVANT[0], bm25=1, vector=9)],
    }
    llm = llm_for({"c1"})
    result = run(llm, FakeSearch(table, default=[]), settings)
    meta = result["relevant"][0].metadata
    assert (meta["bm25_rank"], meta["vector_rank"]) == (1, 2)


def test_broaden_is_told_what_was_rejected(settings):
    llm = llm_for(set())
    run(llm, FakeSearch({}, default=NOISE), settings)
    first = llm.calls_to(prompts.BROADEN_SYSTEM)[0]
    assert "- lab/doc3.md: Сушка филамента PETG" in first
    assert "- lab/doc4.md: Klipper pressure advance" in first
