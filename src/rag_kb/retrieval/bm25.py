"""Sparse retrieval: BM25 over an in-memory index with a RU/EN, code-aware tokenizer."""

import re
from functools import lru_cache

import snowballstemmer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from rag_kb.indexing.chunkers import search_text

WORD_RE = re.compile(r"\w+", re.UNICODE)
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+")
CYRILLIC_RE = re.compile(r"[а-яё]")

_stemmers = {
    "ru": snowballstemmer.stemmer("russian"),
    "en": snowballstemmer.stemmer("english"),
}


@lru_cache(maxsize=100_000)
def _stem(word: str) -> str:
    lang = "ru" if CYRILLIC_RE.search(word) else "en"
    return _stemmers[lang].stemWord(word)


def tokenize(text: str) -> list[str]:
    """Lower-cased, stemmed words; identifiers are kept whole *and* split into parts.

    ``TOKEN_EXPIRY_HOURS`` -> ``token_expiry_hours``, ``token``, ``expiri``, ``hour``;
    ``maxFeedRate`` -> ``maxfeedrate``, ``max``, ``feed``, ``rate``;
    ``принтера`` -> ``принтер``.
    """
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        lower = word.lower()
        parts = [p for chunk in word.split("_") for p in CAMEL_RE.findall(chunk)]
        if len(parts) > 1:
            tokens.append(lower)  # exact identifier match scores highest
            tokens.extend(_stem(p.lower()) for p in parts)
        else:
            tokens.append(_stem(lower))
    return tokens


class BM25Index:
    """Immutable BM25 index; rebuilt as a whole after each indexing run."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        corpus = [tokenize(search_text(d)) for d in docs]
        self._bm25 = BM25Okapi(corpus) if docs else None

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self._bm25 is None:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.docs[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]


class BM25Retriever(BaseRetriever):
    """LangChain adapter. ``index_provider`` returns the current index (swapped on reindex)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_provider: object
    k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        index: BM25Index = self.index_provider()  # type: ignore[operator]
        return [doc for doc, _ in index.search(query, self.k)]
