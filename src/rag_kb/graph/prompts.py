"""Prompts for the Corrective RAG nodes.

Instructions are in English (small models follow them more reliably); the knowledge base
and the questions are in Russian and English, so prompts ask for both languages where
it matters for search and for the question's language in the answer.
"""

REWRITE_SYSTEM = """\
You turn a user question into search queries for a knowledge base about 3D printing \
(printers, firmware, Klipper, electronics, a print workshop). Documents are in Russian \
and English.
Return JSON only: {"ru": "<query in Russian>", "en": "<query in English>"}
Rules: keep names, numbers, model names and code identifiers exactly as written; \
drop filler words; do not answer the question."""

BROADEN_SYSTEM = """\
A search in a 3D printing knowledge base (Russian and English documents) found too \
little for the user's question. Write broader alternative search queries: use synonyms, \
more general terms, related concepts and the other language. Do not repeat the queries \
already tried.
Return JSON only: {"queries": ["<query 1>", "<query 2>", "<query 3>"]}"""

BROADEN_USER = """\
Question: {question}
Already tried: {tried}"""

GRADE_SYSTEM = """\
You check whether a document fragment helps answer a question. The fragment is \
relevant if it contains facts needed for the answer, even partially or in another \
language. It is not relevant if it is only on the same general topic.
Return JSON only: {"relevant": "yes"} or {"relevant": "no"}"""

GRADE_USER = """\
Question: {question}

Fragment ({source}):
{text}"""

GENERATE_SYSTEM = """\
You answer questions using only the numbered context fragments from a knowledge base.
Rules:
- Use only facts from the context; never add outside knowledge.
- Cite fragments in square brackets after the facts they support, e.g. [1] or [2][3].
- Keep exact numbers, names and identifiers as they appear in the context.
- If the context does not contain the answer, say so plainly.
- Answer in {language}. Be concise."""

GENERATE_USER = """\
Context:
{context}

Question: {question}"""

NOT_FOUND = {
    "Russian": "В базе знаний не нашлось информации для ответа на этот вопрос.",
    "English": "The knowledge base does not contain information to answer this question.",
}
