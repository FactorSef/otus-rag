"""Discovering and reading source files."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Tried in order; the first that decodes without errors wins.
ENCODINGS = ("utf-8-sig", "cp1251")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    source: str  # stable identifier stored in chunk metadata
    file_type: str  # extension without the dot, e.g. "md"
    text: str


@dataclass
class ScanResult:
    files: list[SourceFile] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def source_id(path: Path, base: Path | None = None) -> str:
    """Identifier of a file in the index.

    Paths under ``base`` (the working directory by default) are stored relative to it,
    e.g. ``sample_docs/wiki/a8.md``; anything else is stored as an absolute POSIX path.
    """
    base = (base or Path.cwd()).resolve()
    path = path.resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: Path) -> str | None:
    """Read a text file, returning ``None`` for binary or undecodable content."""
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        return None
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        # cp1251 decodes almost any byte sequence: reject results full of control chars.
        return None if _looks_binary(text) else text
    return None


def _looks_binary(text: str, threshold: float = 0.01) -> bool:
    sample = text[:8192]
    allowed = {9, 10, 12, 13}  # tab, newline, form feed, carriage return
    controls = sum(1 for ch in sample if ord(ch) < 32 and ord(ch) not in allowed)
    return bool(sample) and controls / len(sample) > threshold


def _normalize_extensions(extensions: Iterable[str]) -> set[str]:
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}


def scan_folder(
    root: Path,
    pattern: str,
    extensions: Iterable[str],
    max_file_size_kb: int,
    base: Path | None = None,
) -> ScanResult:
    """Collect readable files under ``root`` matching ``pattern`` and ``extensions``."""
    allowed = _normalize_extensions(extensions)
    result = ScanResult()
    max_bytes = max_file_size_kb * 1024

    for path in sorted(p for p in root.glob(pattern) if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue  # hidden files and folders (.git, .venv, ...)
        if path.suffix.lower() not in allowed:
            result.skipped.append((rel, f"unsupported extension '{path.suffix}'"))
            continue
        if path.stat().st_size > max_bytes:
            result.skipped.append((rel, f"larger than {max_file_size_kb} KB"))
            continue
        text = read_text(path)
        if text is None:
            result.skipped.append((rel, "binary or unknown encoding"))
            continue
        if not text.strip():
            result.skipped.append((rel, "empty"))
            continue
        result.files.append(
            SourceFile(
                path=path,
                source=source_id(path, base),
                file_type=path.suffix.lower().lstrip("."),
                text=text,
            )
        )

    logger.info("Scanned %s: %d files, %d skipped", root, len(result.files), len(result.skipped))
    return result
