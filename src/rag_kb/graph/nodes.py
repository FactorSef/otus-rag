"""Nodes of the Corrective RAG graph and the routing decision after grading."""

import json
import logging
import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag_kb.config import Settings
from rag_kb.graph import prompts
from rag_kb.graph.state import RAGState

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[Document]]  # (query, k) -> ranked chunks

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
YES_NO_RE = re.compile(r"\b(yes|no|да|нет)\b", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
MAX_BROADEN_QUERIES = 3


def strip_thinking(text: str) -> str:
    """Drop ``<think>...</think>`` blocks emitted inline by reasoning models (qwen3...)."""
    return THINK_RE.sub("", text or "").strip()


def reply_text(message: Any) -> str:
    text = getattr(message, "text", None)
    return strip_thinking(text if isinstance(text, str) else str(message.content))


# --- parsing helpers (small models often wrap JSON in prose or code fences) -------------


def parse_json(text: str) -> dict[str, Any] | None:
    match = JSON_OBJECT_RE.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_relevance(text: str) -> bool:
    """``True`` for a "yes" verdict; anything unparseable counts as "no"."""
    data = parse_json(text)
    if data is not None:
        value = str(data.get("relevant", data.get("relevance", ""))).strip().lower()
        if value in {"yes", "no", "true", "false", "да", "нет"}:
            return value in {"yes", "true", "да"}
    match = YES_NO_RE.search(text or "")
    return bool(match) and match.group(1).lower() in {"yes", "да"}


def question_language(question: str) -> str:
    cyr = len(CYRILLIC_RE.findall(question))
    lat = len(LATIN_RE.findall(question))
    return "Russian" if cyr >= lat and cyr > 0 else "English"


def query_key(query: str) -> str:
    """Comparison key for queries: case and punctuation do not make a query new."""
    return " ".join(re.findall(r"\w+", query.lower()))


def unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = query_key(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def source_ref(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source', '?')}:{m.get('start_line', '?')}-{m.get('end_line', '?')}"


# --- nodes ------------------------------------------------------------------------------


class Nodes:
    def __init__(
        self,
        llm: BaseChatModel,
        json_llm: BaseChatModel,
        search: SearchFn,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.json_llm = json_llm
        self.search = search
        self.s = settings

    def _messages(self, system: str, user: str) -> list:
        # LLM_PROMPT_SUFFIX: model-specific switches, e.g. "/no_think" for qwen3.
        suffix = f"\n{self.s.llm_prompt_suffix}" if self.s.llm_prompt_suffix else ""
        return [SystemMessage(system), HumanMessage(user + suffix)]

    def _ask(self, model: BaseChatModel, system: str, user: str) -> str:
        return reply_text(model.invoke(self._messages(system, user)))

    def rewrite(self, state: RAGState) -> dict:
        question = state["question"]
        data = parse_json(self._ask(self.json_llm, prompts.REWRITE_SYSTEM, question)) or {}
        rewritten = [str(v) for k, v in data.items() if k in {"ru", "en"} and v]
        # The original question stays: a small model may garble exact identifiers.
        queries = unique([question, *rewritten])
        return {
            "queries": queries,
            "tried": queries,
            "retries": 0,
            "relevant": [],
            "seen_ids": [],
            "steps": [f"rewrite: {queries}"],
        }

    def retrieve(self, state: RAGState) -> dict:
        """Hybrid search for every query; lists are fused by reciprocal rank."""
        seen = set(state.get("seen_ids", []))
        # Already graded chunks are skipped, so look deeper to still get retrieve_k new ones.
        depth = self.s.retrieve_k + len(seen)
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for query in state["queries"]:
            for rank, doc in enumerate(self.search(query, depth), 1):
                if doc.id in seen:
                    continue
                docs[doc.id] = doc
                scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (self.s.rrf_c + rank)
        best = sorted(scores, key=scores.get, reverse=True)[: self.s.retrieve_k]
        candidates = [docs[i] for i in best]
        return {"candidates": candidates, "steps": [f"retrieve: {len(candidates)} new chunks"]}

    def grade(self, state: RAGState) -> dict:
        question, candidates = state["question"], state.get("candidates", [])
        requests = [
            self._messages(
                prompts.GRADE_SYSTEM,
                prompts.GRADE_USER.format(
                    question=question,
                    source=source_ref(doc),
                    text=doc.page_content[: self.s.grade_max_chars],
                ),
            )
            for doc in candidates
        ]
        replies = (
            self.json_llm.batch(requests, config={"max_concurrency": self.s.grade_concurrency})
            if requests
            else []
        )
        verdicts = [parse_relevance(reply_text(r)) for r in replies]
        newly_relevant = [d for d, ok in zip(candidates, verdicts, strict=True) if ok]
        relevant = [*state.get("relevant", []), *newly_relevant]
        return {
            "relevant": relevant,
            "seen_ids": [*state.get("seen_ids", []), *(d.id for d in candidates)],
            "steps": [
                f"grade: {len(newly_relevant)}/{len(candidates)} relevant (total {len(relevant)})"
            ],
        }

    def broaden(self, state: RAGState) -> dict:
        tried = state.get("tried", state.get("queries", []))
        reply = self._ask(
            self.json_llm,
            prompts.BROADEN_SYSTEM,
            prompts.BROADEN_USER.format(question=state["question"], tried="; ".join(tried)),
        )
        data = parse_json(reply) or {}
        proposed = data.get("queries") if isinstance(data.get("queries"), list) else []
        tried_keys = {query_key(q) for q in tried}
        queries = [q for q in unique([str(q) for q in proposed]) if query_key(q) not in tried_keys]
        if not queries:
            # Nothing new from the model: search the question again. retrieve skips the
            # chunks graded so far and goes deeper, so this still brings new material.
            queries = [state["question"]]
        queries = queries[:MAX_BROADEN_QUERIES]
        retries = state.get("retries", 0) + 1
        return {
            "queries": queries,
            "tried": unique([*tried, *queries]),
            "retries": retries,
            "steps": [f"broaden #{retries}: {queries}"],
        }

    def generate(self, state: RAGState) -> dict:
        question = state["question"]
        language = question_language(question)
        context_docs = self._fit_context(state.get("relevant", []))
        if not context_docs:
            return {
                "answer": prompts.NOT_FOUND[language],
                "sources": [],
                "steps": ["generate: no relevant chunks"],
            }

        context = "\n\n".join(
            f"[{i}] ({source_ref(d)})\n{d.page_content}" for i, d in enumerate(context_docs, 1)
        )
        answer = self._ask(
            self.llm,
            prompts.GENERATE_SYSTEM.format(language=language),
            prompts.GENERATE_USER.format(context=context, question=question),
        )
        sources = [
            {
                "ref": i,
                "source": d.metadata.get("source"),
                "lines": f"{d.metadata.get('start_line')}-{d.metadata.get('end_line')}",
                "section": d.metadata.get("section", ""),
            }
            for i, d in enumerate(context_docs, 1)
        ]
        return {
            "answer": answer.strip(),
            "sources": sources,
            "steps": [f"generate: {len(context_docs)} chunks in context"],
        }

    def _fit_context(self, docs: list[Document]) -> list[Document]:
        out, used = [], 0
        for doc in docs:
            if out and used + len(doc.page_content) > self.s.max_context_chars:
                break
            out.append(doc)
            used += len(doc.page_content)
        return out

    # --- routing ----------------------------------------------------------------------

    def route_after_grade(self, state: RAGState) -> Literal["generate", "broaden"]:
        if len(state.get("relevant", [])) >= self.s.min_relevant:
            return "generate"
        if state.get("retries", 0) < self.s.max_retries:
            return "broaden"
        return "generate"  # retry budget spent: answer with what we have (or say not found)
