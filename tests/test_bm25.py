from langchain_core.documents import Document

from rag_kb.retrieval.bm25 import BM25Index, tokenize


def doc(i: int, text: str) -> Document:
    return Document(id=str(i), page_content=text, metadata={"source": f"f{i}.md"})


def test_tokenize_identifiers_kept_whole_and_split():
    tokens = tokenize("TOKEN_EXPIRY_HOURS = 24; maxFeedRate")
    assert "token_expiry_hours" in tokens
    assert {"token", "hour"} <= set(tokens)
    assert {"maxfeedrate", "max", "feed", "rate"} <= set(tokens)


def test_tokenize_stems_russian_and_english():
    assert tokenize("принтера принтеру") == ["принтер", "принтер"]
    assert tokenize("printers printing") == ["printer", "print"]


def test_exact_identifier_ranks_first():
    index = BM25Index(
        [
            doc(1, "Token handling and expiry of sessions, hours of work."),
            doc(2, "TOKEN_EXPIRY_HOURS = 24 configured in auth settings."),
            doc(3, "Bed leveling with a BLTouch probe."),
        ]
    )
    hits = index.search("where is TOKEN_EXPIRY_HOURS used", k=3)
    assert hits[0][0].id == "2"
    assert all(h[0].id != "3" for h in hits)  # no shared terms -> not returned


def test_russian_morphology_matches():
    # BM25Okapi IDF is zero for a term found in half of a two-document corpus,
    # so the corpus needs a few more documents.
    index = BM25Index(
        [
            doc(1, "Калибровка стола принтера"),
            doc(2, "Сушка филамента"),
            doc(3, "Замена сопла"),
            doc(4, "Настройка прошивки"),
        ]
    )
    hits = index.search("как откалибровать стол", k=2)
    assert hits[0][0].id == "1"


def test_empty_index():
    assert BM25Index([]).search("anything", k=5) == []
