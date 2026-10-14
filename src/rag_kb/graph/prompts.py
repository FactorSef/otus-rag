"""Prompts for the Corrective RAG nodes.

Instructions are in English (small models follow them more reliably); the knowledge base
and the questions are in Russian and English, so prompts ask for both languages where
it matters for search and for the question's language in the answer.

Any change here changes the system's behaviour: run ``eval/run_eval.py`` against the
default model afterwards (see CLAUDE.md).
"""

REWRITE_SYSTEM = """\
You turn a user question into search queries for a knowledge base about 3D printing \
(printers, firmware, Klipper, electronics, a print workshop). Documents are in Russian \
and English.
Return JSON only: {"ru": "<query in Russian>", "en": "<query in English>"}
Rules: keep names, numbers, model names and code identifiers exactly as written; \
a proper name in quotes is either kept in its original alphabet or translated as a \
whole, never transliterated letter by letter; drop filler words; do not answer the \
question."""

BROADEN_SYSTEM = """\
A search in a 3D printing knowledge base (Russian and English documents) found too \
little for the user's question. Write broader alternative search queries: use synonyms, \
more general terms, related concepts, likely code or config identifiers and the other \
language. Do not repeat the queries already tried. The rejected fragments were found but \
do not answer the question: steer away from their topics. When the part of the \
question that is still missing is given, aim every query at that part.
Write queries in both languages: documents in the other language are found only by \
queries in that language.
Return JSON only: {"ru": ["<query in Russian>", "<query in Russian>"], \
"en": ["<query in English>", "<query in English>"]}"""

BROADEN_USER = """\
Question: {question}
Still missing: {missing}
Already tried: {tried}
Rejected fragments (file, section, beginning):
{rejected}"""

# Quote first, verdict second: a small model that has to point at the answer in the text
# before deciding grades far more accurately than one asked for a bare yes/no
# (qwen2.5:3b rejected the right chunk with the bare verdict and accepted it this way).
GRADE_SYSTEM = """\
You check whether a document fragment contains the answer to a question.
First copy from the fragment the exact words that answer the question: a sentence, a \
table row, a line of code, a config key with its value, a log line. Then give the verdict.
The fragment is relevant if it contains a fact that answers the question, fully or \
partially. The fact may be in another language than the question (Russian or English) \
and in any format: prose, code, YAML, JSON, logs. Names and terms may be translated, \
transliterated or written as code identifiers. The search queries are the same question \
rephrased; use them to match terms.
If nothing in the fragment answers the question, write "none" as the quote and answer "no".
Return JSON only: {"quote": "<exact words from the fragment, or none>", \
"relevant": "yes" or "no"}"""

GRADE_USER = """\
Question: {question}
Search queries: {queries}

Fragment ({source}):
{text}"""

# Per-part quotes, verified in code (as in the grader): asked for a bare complete/missing
# verdict, qwen3-4b-instruct missed a missing print temperature and flagged answered parts
# as missing; quoting the answer of each part fixed both (10/12 vs 9/12 labelled sets).
# Do not list kinds of parts ("a value, a date, a reason"): the model then returns that
# list as the parts (7/12).
CHECK_SYSTEM = """\
You check whether document fragments answer every part of a question.
First split the question into its parts (for example "when" and "who", or "print \
temperature" and "drying temperature"). For each part copy the exact words from the \
fragments that answer it, or "none" if no fragment answers it. Facts may be in another \
language than the question and in any format (prose, tables, JSON, YAML, code).
Return JSON only: {"parts": [{"part": "<part>", "answer": "<exact words or none>"}]}"""

CHECK_USER = """\
Question: {question}

Fragments:
{fragments}"""

GENERATE_SYSTEM = """\
You answer questions using only the numbered context fragments from a knowledge base.
Rules:
- Use only facts from the context; never add outside knowledge.
- Cite the fragment number in square brackets after each fact it supports, e.g. [1].
- Keep exact numbers, names and identifiers as they appear in the context.
- If the context answers only part of the question, give that part and say plainly \
which part is not in the context. Never fill a missing value with a different value \
from the context.
- If the context does not contain the answer, say so plainly.
- Answer in {language}. Be concise."""

GENERATE_USER = """\
Context:
{context}

Question: {question}"""

LOW_CONFIDENCE_NOTE = {
    "Russian": "Оценка релевантности не подтвердила найденные фрагменты: ответ построен по "
    "лучшим результатам поиска, проверьте источники.",
    "English": "Relevance grading did not confirm the retrieved fragments: the answer is "
    "based on the top search results, check the sources.",
}

# Used when the generated answer has no content besides citations; followed by the
# grader's verified quotes, if any.
ANSWER_FAILED = {
    "Russian": "Не удалось сформулировать ответ по найденным фрагментам, см. источники.",
    "English": "Could not phrase an answer from the retrieved fragments, see the sources.",
}

NOT_FOUND = {
    "Russian": "В базе знаний не нашлось информации для ответа на этот вопрос.",
    "English": "The knowledge base does not contain information to answer this question.",
}
