"""Splitting files into chunks.

The strategy depends on the nature of the file:

* Markdown — split by headings first (a wiki section is the natural unit of meaning),
  then oversized sections are split recursively. The heading breadcrumb is kept in
  metadata and used as context for search.
* Code — ``RecursiveCharacterTextSplitter.from_language`` cuts on class/function
  boundaries first, so a chunk tends to hold a whole definition.
* JSON / YAML — split by structure (key subtrees); the key path becomes the section.
* Plain text — recursive split on paragraphs, then lines, then sentences.

Every chunk carries ``source``, ``file_type``, ``chunk_index``, ``start_line``,
``end_line`` (1-based, inclusive) and ``section``.
"""

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from rag_kb.indexing.loaders import SourceFile

# Bump when the splitting logic changes, so existing indexes get re-chunked.
CHUNKER_VERSION = "2"
CODE_LANGUAGES = {"py": Language.PYTHON, "js": Language.JS, "ts": Language.TS}
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
YAML_TOP_KEY_RE = re.compile(r"^([^\s#\-][^:]*):(\s|$)")


@dataclass(frozen=True)
class Piece:
    """A chunk of text located in the original file."""

    text: str
    start: int  # character offset in the file
    section: str = ""
    end: int | None = None  # exclusive end offset when ``text`` is not verbatim (JSON)


def chunk_id(source: str, index: int) -> str:
    return hashlib.sha1(f"{source}:{index}".encode()).hexdigest()


def search_text(doc: Document) -> str:
    """Text used for embeddings and BM25: chunk content plus its location context."""
    meta = doc.metadata
    header = meta.get("source", "")
    if meta.get("section"):
        header += f" | {meta['section']}"
    return f"{header}\n{doc.page_content}"


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            add_start_index=True,
        )
        self._markdown = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self._code = {
            ext: RecursiveCharacterTextSplitter.from_language(
                lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
            )
            for ext, lang in CODE_LANGUAGES.items()
        }
        self._strategies: dict[str, Callable[[str], list[Piece]]] = {
            "md": self._split_markdown,
            "txt": lambda text: self._split_with(self._text, text),
            "json": self._split_json,
            "yaml": self._split_yaml,
            "yml": self._split_yaml,
            **{ext: (lambda text, s=s: self._split_with(s, text)) for ext, s in self._code.items()},
        }

    @property
    def config_id(self) -> str:
        """Identifies the chunking: when it changes, unchanged files must be re-chunked."""
        return f"{CHUNKER_VERSION}:{self.chunk_size}:{self.chunk_overlap}"

    @property
    def supported_types(self) -> set[str]:
        return set(self._strategies)

    def split(self, file: SourceFile) -> list[Document]:
        strategy = self._strategies.get(file.file_type, self._strategies["txt"])
        pieces = [p for p in strategy(file.text) if p.text.strip()]
        line_starts = _line_starts(file.text)
        docs = []
        for index, piece in enumerate(pieces):
            if piece.end is None:
                # page_content is stripped: point the line range at the text itself,
                # not at leading/trailing blank lines of the piece.
                start = piece.start + len(piece.text) - len(piece.text.lstrip())
                end = piece.start + len(piece.text.rstrip())
            else:  # re-serialized JSON: the span in the file is given explicitly
                start, end = piece.start, piece.end
            start_line = _line_of(line_starts, start)
            end_line = _line_of(line_starts, max(end - 1, start))
            docs.append(
                Document(
                    id=chunk_id(file.source, index),
                    page_content=piece.text.strip(),
                    metadata={
                        "source": file.source,
                        "file_type": file.file_type,
                        "chunk_index": index,
                        "start_line": start_line,
                        "end_line": max(end_line, start_line),
                        "section": piece.section,
                    },
                )
            )
        return docs

    # --- strategies -------------------------------------------------------------------

    @staticmethod
    def _split_with(
        splitter: RecursiveCharacterTextSplitter, text: str, offset: int = 0, section: str = ""
    ) -> list[Piece]:
        return [
            Piece(d.page_content, offset + max(d.metadata.get("start_index", 0), 0), section)
            for d in splitter.create_documents([text])
        ]

    def _split_markdown(self, text: str) -> list[Piece]:
        pieces = []
        for start, body, breadcrumb in _markdown_sections(text):
            section = " > ".join(breadcrumb)
            if all(HEADING_RE.match(ln) for ln in body.splitlines() if ln.strip()):
                continue  # heading with no own content: its title lives in children's section
            if len(body) <= self.chunk_size:
                pieces.append(Piece(body, start, section))
            else:
                pieces.extend(self._split_with(self._markdown, body, start, section))
        if not pieces:  # stub page made of headings only: keep it searchable as a whole
            return self._split_with(self._markdown, text)
        return pieces

    def _split_yaml(self, text: str) -> list[Piece]:
        """Split on top-level keys, keeping the original YAML text."""
        try:
            yaml.safe_load(text)
        except yaml.YAMLError:
            return self._split_with(self._text, text)

        lines = text.splitlines(keepends=True)
        blocks: list[tuple[int, str]] = []  # (line index, key)
        for i, line in enumerate(lines):
            m = YAML_TOP_KEY_RE.match(line)
            if m:
                blocks.append((i, m.group(1).strip().strip("'\"")))
        if not blocks:
            return self._split_with(self._text, text)

        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        # Comments / document markers before the first key belong to the first block.
        bounds = [0] + [i for i, _ in blocks[1:]] + [len(lines)]
        pieces = []
        for (_, key), lo, hi in zip(blocks, bounds, bounds[1:], strict=False):
            body = "".join(lines[lo:hi])
            if len(body) <= self.chunk_size:
                pieces.append(Piece(body, offsets[lo], key))
            else:
                pieces.extend(self._split_with(self._text, body, offsets[lo], key))
        return pieces

    def _split_json(self, text: str) -> list[Piece]:
        """Split by key subtrees; each chunk is a valid JSON fragment of the original."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._split_with(self._text, text)
        if not isinstance(data, dict):
            data = {"items": data}

        splitter = RecursiveJsonSplitter(max_chunk_size=self.chunk_size)
        parts = splitter.split_json(data, convert_lists=True)
        line_starts = _line_starts(text)
        # Split on newline only, exactly like _line_starts: str.splitlines() also breaks
        # on CR, NEL, U+2028... and would make line indices disagree with line_starts.
        lines = text.split("\n")

        located = []
        cursor_line = 0
        for part in parts:
            path = _first_path(part)
            cursor_line = _locate_path(lines, path, cursor_line)
            located.append((part, path, line_starts[cursor_line]))

        # Chunks are re-serialized, so their span in the file runs up to the next chunk.
        pieces = []
        for i, (part, path, start) in enumerate(located):
            end = located[i + 1][2] if i + 1 < len(located) else len(text.rstrip())
            body = json.dumps(part, ensure_ascii=False, indent=1)
            pieces.append(Piece(body, start, ".".join(path), end=max(end, start + 1)))
        return pieces


# --- helpers ----------------------------------------------------------------------------


def _markdown_sections(text: str) -> Iterator[tuple[int, str, list[str]]]:
    """Yield ``(offset, section_text, heading_breadcrumb)``; ignores headings in code fences."""
    breadcrumb: list[str] = []
    section_start, section_crumb = 0, []
    offset, in_fence = 0, False
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line.rstrip("\r\n"))
        if m:
            if offset > section_start:
                yield section_start, text[section_start:offset], section_crumb
            level = len(m.group(1))
            breadcrumb = [*breadcrumb[: level - 1], m.group(2)]
            section_start, section_crumb = offset, breadcrumb
        offset += len(line)
    if offset > section_start:
        yield section_start, text[section_start:], section_crumb


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _line_of(line_starts: list[int], offset: int) -> int:
    """1-based line number containing ``offset`` (binary search)."""
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _first_path(part: dict) -> list[str]:
    path = []
    node = part
    while isinstance(node, dict) and node:
        key = next(iter(node))
        path.append(str(key))
        node = node[key]
    return path


def _locate_path(lines: list[str], path: list[str], start: int) -> int:
    """Find the line of the deepest locatable key of ``path`` at or after ``start``."""
    line = start
    for key in path:
        if key.isdigit() or key == "items":
            continue  # list indices (from convert_lists) have no textual key
        needle = json.dumps(key, ensure_ascii=False)
        for i in range(line, len(lines)):
            if needle in lines[i]:
                line = i
                break
    return line
