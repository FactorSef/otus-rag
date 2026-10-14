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
NeighborFn = Callable[[Document], list[Document]]  # chunk -> adjacent chunks of its file

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
YES_NO_RE = re.compile(r"\b(yes|no|да|нет)\b", re.IGNORECASE)
# The verdict field of a grader reply whose JSON is broken (e.g. an unescaped quote).
RELEVANT_FIELD_RE = re.compile(r'"relevan\w*"\s*:\s*"?(yes|no|true|false|да|нет)\b', re.IGNORECASE)
QUOTE_FIELD_RE = re.compile(r'"quote"\s*:\s*"(.*?)"\s*,\s*"relevan', re.DOTALL)
CITATION_RE = re.compile(r"\[(\d+)\]")
WORD_RE = re.compile(r"\w{2,}")
THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
MAX_BROADEN_QUERIES = 3
FALLBACK_CHUNKS = 3  # rejected top-ranked chunks used for a low-confidence answer
RANK_KEYS = ("bm25_rank", "vector_rank")  # per-list ranks set by the search function


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


def parse_grade(text: str) -> tuple[bool, str]:
    """Grader reply -> (verdict, quote). Only the verdict field decides; unparseable is "no"."""
    yes = {"yes", "true", "да"}
    text = text or ""
    data = parse_json(text)
    if data is not None:
        value = str(data.get("relevant", data.get("relevance", ""))).strip().lower()
        if value in {"yes", "no", "true", "false", "да", "нет"}:
            return value in yes, str(data.get("quote", "") or "")
    fields = RELEVANT_FIELD_RE.findall(text)
    if fields:
        quote = QUOTE_FIELD_RE.search(text)
        return fields[-1].lower() in yes, quote.group(1) if quote else ""
    if "{" in text:
        return False, ""  # broken JSON without a verdict: its words may come from a quote
    match = YES_NO_RE.search(text)
    return bool(match) and match.group(1).lower() in yes, ""


def parse_relevance(text: str) -> bool:
    return parse_grade(text)[0]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def verified_quote(quote: str, text: str) -> str:
    """The grader's quote if it really occurs in the chunk (small models invent quotes)."""
    key = normalize_text(quote)
    if not key or key in {"none", "нет"} or key not in normalize_text(text):
        return ""
    return " ".join(quote.split())


def check_citations(answer: str, n_sources: int) -> tuple[str, list[int], bool]:
    """Drop citations of non-existent fragments.

    Returns (answer, dropped numbers, whether anything besides citations is left).
    """
    valid = range(1, n_sources + 1)
    dropped = sorted({int(n) for n in CITATION_RE.findall(answer)} - set(valid))
    if dropped:
        answer = CITATION_RE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)
        answer = re.sub(r"[ \t]+([.,;:])", r"\1", answer).strip()
    has_content = bool(WORD_RE.search(CITATION_RE.sub(" ", answer)))
    return answer, dropped, has_content


def with_metadata(doc: Document, **extra: Any) -> Document:
    """Copy of ``doc`` with extra metadata (documents may be shared with the index)."""
    return Document(id=doc.id, page_content=doc.page_content, metadata={**doc.metadata, **extra})


def strong_rank(doc: Document, limit: int) -> bool:
    """Ranked within ``limit`` in both the BM25 and the vector list."""
    ranks = [doc.metadata.get(key) for key in RANK_KEYS]
    return all(r is not None and r <= limit for r in ranks)


def mixed_script(query: str) -> bool:
    """A word mixing Cyrillic and Latin letters ("Dымчатый"): a garbled transliteration.

    Such a query matches nothing in BM25 and misleads the grader, so it is dropped.
    """
    return any(
        CYRILLIC_RE.search(word) and LATIN_RE.search(word)
        for word in re.findall(r"[^\W\d_]+", query)
    )


def parse_check(text: str, fragments: str) -> list[str]:
    """Completeness reply -> the parts of the question left unanswered.

    A part counts as answered only if its quoted answer really occurs in the fragments
    (compared word by word, like the grader's quote). An unparseable reply means
    "nothing missing": a broken check must not cost an extra search round.
    """
    data = parse_json(text)
    parts = data.get("parts") if data else None
    if not isinstance(parts, list):
        return []
    haystack = normalize_text(fragments)
    missing = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        answer = normalize_text(str(part.get("answer", "") or ""))
        if not answer or answer in {"none", "нет"} or answer not in haystack:
            missing.append(str(part.get("part", "") or "?").strip())
    return missing


def interleave(*lists: list) -> list:
    """[a1, a2], [b1, b2] -> [a1, b1, a2, b2]: a truncated list keeps both languages."""
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out += [x[i] for x in lists if i < len(x)]
    return out


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


def unique_ids(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def source_ref(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source', '?')}:{m.get('start_line', '?')}-{m.get('end_line', '?')}"


def rejected_line(doc: Document, chars: int = 100) -> str:
    section = doc.metadata.get("section") or ""
    start = " ".join(doc.page_content.split())[:chars]
    return (
        f"- {doc.metadata.get('source', '?')}" + (f" [{section}]" if section else "") + f": {start}"
    )


# --- nodes ------------------------------------------------------------------------------


class Nodes:
    def __init__(
        self,
        llm: BaseChatModel,
        json_llm: BaseChatModel,
        search: SearchFn,
        settings: Settings,
        neighbors: NeighborFn | None = None,
    ) -> None:
        self.llm = llm
        self.json_llm = json_llm
        self.search = search
        self.neighbors = neighbors
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
        garbled = [q for q in rewritten if mixed_script(q)]
        # The original question stays: a small model may garble exact identifiers.
        queries = unique([question, *(q for q in rewritten if q not in garbled)])
        step = f"rewrite: {queries}"
        if garbled:
            step += f" (dropped mixed-alphabet {garbled})"
        return {
            "queries": queries,
            "rewritten": queries[1:],
            "tried": queries,
            "retries": 0,
            "relevant": [],
            "rejected": [],
            "seen_ids": [],
            "missing": "",
            "steps": [step],
        }

    def retrieve(self, state: RAGState) -> dict:
        """Hybrid search for every query; lists are fused by reciprocal rank."""
        seen = set(state.get("seen_ids", []))
        # Already graded chunks are skipped, so look deeper to still get retrieve_k new ones.
        depth = self.s.retrieve_k + len(seen)
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        ranks: dict[str, dict[str, int]] = {}  # best BM25 / vector rank over all queries
        firsts: list[str] = []  # best new chunk of every query
        for query in state["queries"]:
            first = None
            for rank, doc in enumerate(self.search(query, depth), 1):
                if doc.id in seen:
                    continue
                first = first or doc.id
                docs[doc.id] = doc
                scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (self.s.rrf_c + rank)
                best_ranks = ranks.setdefault(doc.id, {})
                for key in RANK_KEYS:
                    value = doc.metadata.get(key)
                    if value is not None:
                        best_ranks[key] = min(value, best_ranks.get(key, value))
            if first and first not in firsts:
                firsts.append(first)
        # Every query's top hit is kept before fusion fills the rest: summed RRF lets two
        # similar Russian queries outvote the only English one, whose rank-1 hit (the
        # English file with the answer) then never reached the grader.
        fused = sorted(scores, key=scores.get, reverse=True)
        best = unique_ids([*firsts, *fused])[: self.s.retrieve_k]
        candidates = [with_metadata(docs[i], **ranks[i]) for i in best]
        return {"candidates": candidates, "steps": [f"retrieve: {len(candidates)} new chunks"]}

    def grade(self, state: RAGState) -> dict:
        question, candidates = state["question"], state.get("candidates", [])
        # The RU/EN rewrites carry the question's terms into the other language: a small
        # model cannot match a Russian question to an English YAML key on its own.
        queries = "; ".join(state.get("rewritten") or [question])
        requests = [
            self._messages(
                prompts.GRADE_SYSTEM,
                prompts.GRADE_USER.format(
                    question=question,
                    queries=queries,
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
        newly_relevant, rejected = [], []
        for doc, reply in zip(candidates, replies, strict=True):
            ok, quote = parse_grade(reply_text(reply))
            if ok:
                quote = verified_quote(quote, doc.page_content)
                newly_relevant.append(with_metadata(doc, grade_quote=quote) if quote else doc)
            else:
                rejected.append(doc)
        relevant = [*state.get("relevant", []), *newly_relevant]
        step = f"grade: {len(newly_relevant)}/{len(candidates)} relevant (total {len(relevant)})"
        if newly_relevant:
            step += ": " + ", ".join(source_ref(d) for d in newly_relevant)
        return {
            "relevant": relevant,
            # Rejected chunks are not graded again (the grader is deterministic) but stay
            # available to generate as a low-confidence fallback.
            "rejected": [*state.get("rejected", []), *rejected],
            "seen_ids": [*state.get("seen_ids", []), *(d.id for d in candidates)],
            "steps": [step],
        }

    def broaden(self, state: RAGState) -> dict:
        tried = state.get("tried", state.get("queries", []))
        relevant_ids = {d.id for d in state.get("relevant", [])}
        last_rejected = [d for d in state.get("candidates", []) if d.id not in relevant_ids]
        reply = self._ask(
            self.json_llm,
            prompts.BROADEN_SYSTEM,
            prompts.BROADEN_USER.format(
                question=state["question"],
                missing=state.get("missing") or "(unknown)",
                tried="; ".join(tried),
                rejected="\n".join(rejected_line(d) for d in last_rejected) or "- (none)",
            ),
        )
        data = parse_json(reply) or {}
        lists = [data.get(key) for key in ("ru", "en", "queries")]
        proposed = interleave(*(x for x in lists if isinstance(x, list)))
        tried_keys = {query_key(q) for q in tried}
        queries = [
            q
            for q in unique([str(q) for q in proposed])
            if query_key(q) not in tried_keys and not mixed_script(q)
        ]
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

    def check(self, state: RAGState) -> dict:
        """Do the accepted facts answer every part of the question?

        Enough relevant chunks is not enough: "print and dry temperature" with only the
        drying table found would otherwise stop the search and get a half answer.
        """
        # Full chunk text, not the grader's quote: a quote is one line of the chunk, and
        # the check then reported answered parts (in the rest of the chunk) as missing.
        docs = self._fit_context(state.get("relevant", []))
        fragments = "\n\n".join(
            f"({source_ref(d)})\n{d.page_content[: self.s.grade_max_chars]}" for d in docs
        )
        reply = self._ask(
            self.json_llm,
            prompts.CHECK_SYSTEM,
            prompts.CHECK_USER.format(question=state["question"], fragments=fragments),
        )
        missing = "; ".join(parse_check(reply, fragments))
        if not missing:
            return {"missing": "", "steps": ["check: complete"]}
        return {"missing": missing, "steps": [f"check: missing {missing!r}"]}

    def generate(self, state: RAGState) -> dict:
        question = state["question"]
        language = question_language(question)
        steps = []
        context_docs = self._fit_context(state.get("relevant", []))
        low_confidence = False
        if not context_docs:
            context_docs = self._fallback(state.get("rejected", []))
            low_confidence = bool(context_docs)
            if low_confidence:
                steps.append(
                    f"generate: grader accepted nothing, using {len(context_docs)} chunks "
                    f"ranked in the top {self.s.fallback_max_rank} of both BM25 and vector "
                    "search (low confidence)"
                )
        if not context_docs:
            return {
                "answer": prompts.NOT_FOUND[language],
                "sources": [],
                "low_confidence": False,
                "steps": ["generate: no relevant chunks"],
            }
        context_docs, added = self._with_neighbors(context_docs)
        if added:
            steps.append(f"generate: +{added} adjacent chunks of the same files")

        context = "\n\n".join(
            f"[{i}] ({source_ref(d)})\n{d.page_content}" for i, d in enumerate(context_docs, 1)
        )
        answer = self._ask(
            self.llm,
            prompts.GENERATE_SYSTEM.format(language=language),
            prompts.GENERATE_USER.format(context=context, question=question),
        )
        answer, dropped, has_content = check_citations(answer.strip(), len(context_docs))
        if dropped:
            steps.append(f"generate: dropped citations of missing fragments {dropped}")
        if not has_content:
            # e.g. a bare "[2]": a 3B model sometimes answers with a citation only.
            steps.append("generate: answer had no content besides citations")
            answer = self._quotes_answer(context_docs, language)
        if low_confidence:
            answer = f"{prompts.LOW_CONFIDENCE_NOTE[language]}\n\n{answer}"
        sources = [
            {
                "ref": i,
                "source": d.metadata.get("source"),
                "lines": f"{d.metadata.get('start_line')}-{d.metadata.get('end_line')}",
                "section": d.metadata.get("section", ""),
                **({"adjacent": True} if d.metadata.get("adjacent") else {}),
            }
            for i, d in enumerate(context_docs, 1)
        ]
        return {
            "answer": answer,
            "sources": sources,
            "low_confidence": low_confidence,
            "steps": [*steps, f"generate: {len(context_docs)} chunks in context"],
        }

    def _fallback(self, rejected: list[Document]) -> list[Document]:
        """Rejected chunks that both retrievers ranked high: the grader may be wrong."""
        limit = self.s.fallback_max_rank
        if not limit:
            return []
        strong = [d for d in rejected if strong_rank(d, limit)]
        strong.sort(key=lambda d: sum(d.metadata[key] for key in RANK_KEYS))
        return self._fit_context(strong[:FALLBACK_CHUNKS])

    @staticmethod
    def _quotes_answer(docs: list[Document], language: str) -> str:
        """Answer from the grader's verified quotes when generation produced nothing."""
        lines = [prompts.ANSWER_FAILED[language]]
        lines += [
            f"- «{d.metadata['grade_quote']}» [{i}]"
            for i, d in enumerate(docs, 1)
            if d.metadata.get("grade_quote")
        ]
        return "\n".join(lines)

    def _with_neighbors(self, docs: list[Document]) -> tuple[list[Document], int]:
        """Add adjacent chunks of the same files while the context budget allows.

        Graded chunks keep priority (they are already in ``docs``); each neighbour is
        placed next to its chunk so a file reads in order. Returns (context, added).
        """
        if not self.neighbors or not self.s.context_neighbors:
            return docs, 0
        used = sum(len(d.page_content) for d in docs)
        taken = {d.id for d in docs}
        before: dict[str, list[Document]] = {}
        after: dict[str, list[Document]] = {}
        for doc in docs:
            index = doc.metadata.get("chunk_index", 0)
            for near in self.neighbors(doc):
                if near.id in taken or used + len(near.page_content) > self.s.max_context_chars:
                    continue
                taken.add(near.id)
                used += len(near.page_content)
                side = before if near.metadata.get("chunk_index", 0) < index else after
                side.setdefault(doc.id, []).append(with_metadata(near, adjacent=True))
        out = []
        for doc in docs:
            out += [*before.get(doc.id, []), doc, *after.get(doc.id, [])]
        return out, len(out) - len(docs)

    def _fit_context(self, docs: list[Document]) -> list[Document]:
        out, used = [], 0
        for doc in docs:
            if out and used + len(doc.page_content) > self.s.max_context_chars:
                break
            out.append(doc)
            used += len(doc.page_content)
        return out

    # --- routing ----------------------------------------------------------------------

    def route_after_grade(self, state: RAGState) -> Literal["check", "generate", "broaden"]:
        can_retry = state.get("retries", 0) < self.s.max_retries
        if len(state.get("relevant", [])) >= self.s.min_relevant:
            # The check only matters if a missing part could still be searched for.
            return "check" if self.s.completeness_check and can_retry else "generate"
        if can_retry:
            return "broaden"
        return "generate"  # retry budget spent: answer with what we have (or say not found)

    def route_after_check(self, state: RAGState) -> Literal["generate", "broaden"]:
        return "broaden" if state.get("missing") else "generate"
