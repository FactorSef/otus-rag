"""State of the Corrective RAG graph."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str
    queries: list[str]  # search queries of the current iteration
    tried: list[str]  # every query searched so far (shown to broaden, never repeated)
    candidates: list[Document]  # chunks retrieved in this iteration, not graded yet
    relevant: list[Document]  # graded relevant, accumulated across iterations
    seen_ids: list[str]  # every chunk graded so far (never re-graded)
    retries: int  # broaden -> retrieve loops done
    answer: str
    sources: list[dict]
    steps: Annotated[list[str], operator.add]  # execution trace, appended by every node
