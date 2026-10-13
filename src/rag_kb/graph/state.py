"""State of the Corrective RAG graph."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str
    queries: list[str]  # search queries of the current iteration
    rewritten: list[str]  # RU/EN rewrites of the question (shown to the grader)
    tried: list[str]  # every query searched so far (shown to broaden, never repeated)
    candidates: list[Document]  # chunks retrieved in this iteration, not graded yet
    relevant: list[Document]  # graded relevant, accumulated across iterations
    rejected: list[Document]  # graded not relevant, kept for the low-confidence fallback
    seen_ids: list[str]  # every chunk graded so far (never re-graded)
    retries: int  # broaden -> retrieve loops done
    answer: str
    sources: list[dict]
    low_confidence: bool  # answer built from rejected but top-ranked chunks
    steps: Annotated[list[str], operator.add]  # execution trace, appended by every node
