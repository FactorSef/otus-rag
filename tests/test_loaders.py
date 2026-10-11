from pathlib import Path

from rag_kb.config import DEFAULT_EXTENSIONS
from rag_kb.indexing.loaders import read_text, scan_folder, source_id


def test_scan_filters_unsupported_hidden_binary_and_empty(docs_dir: Path):
    (docs_dir / "image.png").write_bytes(b"\x89PNG\x00\x00")
    (docs_dir / "blob.txt").write_bytes(b"abc\x00def")
    (docs_dir / "empty.md").write_text("   \n", encoding="utf-8")
    (docs_dir / ".git").mkdir()
    (docs_dir / ".git" / "config.yaml").write_text("a: 1", encoding="utf-8")

    result = scan_folder(docs_dir, "**/*", DEFAULT_EXTENSIONS, max_file_size_kb=64)

    types = sorted(f.file_type for f in result.files)
    assert types == ["js", "json", "md", "py", "ts", "txt", "yaml"]
    skipped = dict(result.skipped)
    assert "unsupported" in skipped["image.png"]
    assert "binary" in skipped["blob.txt"]
    assert skipped["empty.md"] == "empty"
    assert not any(".git" in f.source for f in result.files)


def test_glob_pattern_and_size_limit(docs_dir: Path):
    (docs_dir / "big.md").write_text("x" * 3000, encoding="utf-8")
    result = scan_folder(docs_dir, "**/*.md", DEFAULT_EXTENSIONS, max_file_size_kb=1)
    assert [f.path.name for f in result.files] == ["guide.md"]
    assert dict(result.skipped)["big.md"].startswith("larger than")


def test_cp1251_fallback(tmp_path: Path):
    path = tmp_path / "legacy.txt"
    path.write_bytes("Привет, принтер".encode("cp1251"))
    assert read_text(path) == "Привет, принтер"


def test_source_id_relative_to_base(tmp_path: Path):
    path = tmp_path / "docs" / "a.md"
    assert source_id(path, base=tmp_path) == "docs/a.md"
    other = Path(tmp_path.anchor) / "elsewhere" / "b.md"
    assert source_id(other, base=tmp_path) == other.resolve().as_posix()
