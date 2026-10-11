import json
import textwrap
from pathlib import Path

import pytest

from helpers import SAMPLE_FILES
from rag_kb.indexing.chunkers import Chunker
from rag_kb.indexing.loaders import SourceFile


def make_file(name: str, content: str | None = None) -> SourceFile:
    text = textwrap.dedent(content if content is not None else SAMPLE_FILES[name])
    return SourceFile(path=Path(name), source=name, file_type=name.rsplit(".", 1)[-1], text=text)


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(chunk_size=300, chunk_overlap=40)


def lines_of(file: SourceFile, start: int, end: int) -> str:
    return "\n".join(file.text.splitlines()[start - 1 : end])


@pytest.mark.parametrize("name", list(SAMPLE_FILES))
def test_every_format_produces_chunks_with_metadata(chunker, name):
    file = make_file(name)
    docs = chunker.split(file)
    assert docs, name
    for i, doc in enumerate(docs):
        meta = doc.metadata
        assert meta["source"] == name
        assert meta["chunk_index"] == i
        assert 1 <= meta["start_line"] <= meta["end_line"] <= len(file.text.splitlines())
        assert doc.id and doc.page_content.strip()
    assert len({d.id for d in docs}) == len(docs)


def test_markdown_split_by_headings(chunker):
    file = make_file("guide.md")
    docs = chunker.split(file)
    sections = [d.metadata["section"] for d in docs]
    assert sections == [
        "Klipper guide",
        "Klipper guide > Pressure advance",
        "Klipper guide > Input shaper",
    ]
    # the "# not a heading" line inside a code fence did not start a new section
    pressure = docs[1]
    assert "echo tune" in pressure.page_content
    # line numbers point at the original text
    assert (
        pressure.page_content.strip()
        == lines_of(file, pressure.metadata["start_line"], pressure.metadata["end_line"]).strip()
    )


def test_long_markdown_section_is_split_with_overlap(chunker):
    body = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(12))
    docs = chunker.split(make_file("long.md", f"# Title\n\n## Big\n\n{body}\n"))
    assert len(docs) > 2
    assert all(len(d.page_content) <= 300 for d in docs)
    assert all(d.metadata["section"] == "Title > Big" for d in docs)
    starts = [d.metadata["start_line"] for d in docs]
    assert starts == sorted(starts)


def test_code_split_on_definition_boundaries():
    docs = Chunker(chunk_size=120, chunk_overlap=0).split(make_file("tools/purge.py"))
    first_lines = [d.page_content.splitlines()[0] for d in docs]
    assert first_lines[0] == "PURGE_LINE_LENGTH_MM = 143"
    assert any(line.startswith("def purge_line") for line in first_lines)
    assert any(line.startswith("class Calibrator") for line in first_lines)


def test_json_split_by_keys(chunker):
    file = make_file("data/filaments.json")
    big = {f"item{i}": {"name": f"Filament {i}", "temp": 200 + i} for i in range(20)}
    docs = chunker.split(make_file("big.json", json.dumps(big, indent=2)))
    assert len(docs) > 1
    for doc in docs:
        json.loads(doc.page_content)  # every chunk is valid JSON
        assert doc.metadata["section"].startswith("item")
    # first chunk of the small file starts at the "petg" key line
    first = chunker.split(file)[0]
    assert "Smoky Quartz PETG" in first.page_content
    assert '"petg"' in file.text.splitlines()[first.metadata["start_line"] - 1]


def test_yaml_split_by_top_level_keys():
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    file = make_file("data/fleet.yaml")
    docs = chunker.split(file)
    assert [d.metadata["section"] for d in docs] == ["soyka", "kestrel"]
    assert "# Printer fleet" in docs[0].page_content  # leading comment kept
    assert docs[1].metadata["start_line"] == 5


def test_headings_only_markdown_still_indexed(chunker):
    docs = chunker.split(make_file("stub.md", "# Extruder Breakout Board\n"))
    assert [d.page_content for d in docs] == ["# Extruder Breakout Board"]


def test_invalid_json_falls_back_to_text(chunker):
    docs = chunker.split(make_file("broken.json", '{"a": 1,, oops'))
    assert len(docs) == 1
    assert docs[0].metadata["section"] == ""
