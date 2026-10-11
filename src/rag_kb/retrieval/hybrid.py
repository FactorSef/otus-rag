"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion.

Fusion is done by LangChain's ``EnsembleRetriever`` (weighted RRF:
``score(d) = sum_i w_i / (c + rank_i(d))``). We call its fusion on the two ranked lists
directly so that per-list ranks can be reported to the user as well.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_kb.retrieval.bm25 import BM25Index, BM25Retriever
from rag_kb.retrieval.store import ChromaStore

ID_KEY = "chunk_id"


@dataclass(frozen=True)
class SearchHit:
    doc: Document
    score: float  # fused RRF score
    bm25_rank: int | None  # 1-based position in the BM25 list, None if absent
    vector_rank: int | None


class VectorRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: ChromaStore
    embeddings: Embeddings
    k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        return [doc for doc, _ in self.store.query(vector, self.k)]


class HybridSearch:
    def __init__(
        self,
        store: ChromaStore,
        embeddings: Embeddings,
        bm25_provider: Callable[[], BM25Index],
        bm25_weight: float,
        vector_weight: float,
        rrf_c: int,
        fetch_k: int,
    ) -> None:
        # Each list is fetched deeper than the final top-k so fusion has material to work with.
        self.bm25 = BM25Retriever(index_provider=bm25_provider, k=fetch_k)
        self.vector = VectorRetriever(store=store, embeddings=embeddings, k=fetch_k)
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.vector],
            weights=[bm25_weight, vector_weight],
            c=rrf_c,
            id_key=ID_KEY,
        )

    def search(self, query: str, k: int) -> list[SearchHit]:
        lists = [
            [_with_id(d) for d in self.bm25.invoke(query)],
            [_with_id(d) for d in self.vector.invoke(query)],
        ]
        fused = self.ensemble.weighted_reciprocal_rank(lists)

        ranks = [{d.metadata[ID_KEY]: pos for pos, d in enumerate(lst, 1)} for lst in lists]
        weights, c = self.ensemble.weights, self.ensemble.c
        hits = []
        for doc in fused[:k]:
            key = doc.metadata[ID_KEY]
            per_list = [r.get(key) for r in ranks]
            score = sum(w / (c + r) for w, r in zip(weights, per_list, strict=True) if r)
            hits.append(SearchHit(doc, score, per_list[0], per_list[1]))
        return hits


def _with_id(doc: Document) -> Document:
    """EnsembleRetriever dedups by ``metadata[id_key]``; copy the chunk id there."""
    return Document(
        id=doc.id, page_content=doc.page_content, metadata={**doc.metadata, ID_KEY: doc.id}
    )
