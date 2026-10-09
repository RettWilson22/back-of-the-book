"""Load course materials into pages, keeping the page or slide number of every piece of text.

Citations are only as good as the location data behind them, so every loader returns
`Page` objects tagged with the file name and the 1-indexed page/slide they came from.
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = (".pdf", ".pptx", ".md", ".txt")
# Repeated headers/footers are only stripped from long documents (books). In short documents a
# line on every page is usually real content, like the course name or the teacher's name.
BOILERPLATE_MIN_PAGES = 12
# A .pptx is a zip archive. These limits refuse archives that would expand to far more data
# than any real slide deck before python-pptx opens them.
MAX_ARCHIVE_MEMBERS = 2000
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024  # total uncompressed size
MAX_ARCHIVE_MEMBER_BYTES = 50 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


@dataclass(frozen=True)
class Page:
    source: str  # file name shown in citations, e.g. "lecture3.pdf"
    page: int  # 1-indexed page (PDF), slide (PPTX), or section (MD/TXT) number
    text: str
    label: str | None = None  # printed page number, when it differs from `page`


class UnsupportedFileError(ValueError):
    """A file that won't be read: an unsupported type, over a size limit, or a suspicious
    archive. The message is safe to show to users."""


@dataclass(frozen=True)
class DocumentText:
    pages: list[Page]
    unreadable: list[int] = field(default_factory=list)  # PDF pages with no text (scans)


def _too_much_text(name: str, max_chars: int) -> UnsupportedFileError:
    return UnsupportedFileError(
        f"{name} has too much text to read here (the limit is {max_chars:,} characters). "
        "Try splitting it into smaller files."
    )


def _check_length(name: str, pages: list[Page], max_chars: int | None) -> None:
    if max_chars is not None and sum(len(p.text) for p in pages) > max_chars:
        raise _too_much_text(name, max_chars)


def clean_text(text: str) -> str:
    """Normalize extracted text: rejoin hyphenated line breaks and collapse whitespace."""
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(
        r"(\w) -([a-z])", r"\1-\2", text
    )  # pypdf splits "peer-reviewed" as "peer -reviewed"
    # Two linear passes. A single r"\s*\n\s*" backtracks over every long run of spaces that
    # has no line break in it (such as non-breaking spaces), which is quadratic.
    text = re.sub(r"[^\S\n]+", " ", text)  # any run of spaces, tabs, NBSPs... -> one space
    text = re.sub(r" ?\n\s*", "\n", text)
    return text.strip()


# Running heads such as "10     1 • Chapter title" (even pages) or "1.1 • Section     11" (odd).
_RUNNING_HEAD = re.compile(r"^(\d{1,4})\s{2,}\S.*\u2022|\u2022.*\S\s{2,}(\d{1,4})$")
# Running heads are short. Longer lines are never matched: the pattern is quadratic on a long
# line with many bullets.
_RUNNING_HEAD_MAX_CHARS = 200


def _running_head(line: str) -> re.Match[str] | None:
    line = line.strip()
    return _RUNNING_HEAD.search(line) if len(line) <= _RUNNING_HEAD_MAX_CHARS else None


def detect_page_offset(raw_pages: list[str], min_share: float = 0.3) -> int | None:
    """Return `pdf_page - printed_page` if most running heads agree on one offset."""
    votes: Counter[int] = Counter()
    for pdf_page, text in enumerate(raw_pages, start=1):
        for line in text.splitlines():
            match = _running_head(line)
            if match:
                votes[pdf_page - int(match.group(1) or match.group(2))] += 1
                break
    if not votes:
        return None
    offset, count = votes.most_common(1)[0]
    return offset if count >= max(3, min_share * len(raw_pages)) else None


def remove_boilerplate(raw_pages: list[str], min_share: float = 0.2) -> list[str]:
    """Drop running heads and lines repeated on many pages (e.g. "Access for free at ...")."""
    counts = Counter(
        line.strip() for text in raw_pages for line in set(text.splitlines()) if line.strip()
    )
    threshold = max(3, min_share * len(raw_pages))
    repeated = {line for line, n in counts.items() if n >= threshold}
    return [
        "\n".join(
            line
            for line in text.splitlines()
            if line.strip() not in repeated and not _running_head(line)
        )
        for text in raw_pages
    ]


def read_pdf(
    path: Path, max_pages: int | None = None, max_chars: int | None = None
) -> DocumentText:
    """Read a PDF once: its pages, plus the pages with no extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    count = len(reader.pages)
    if max_pages is not None and count > max_pages:
        raise UnsupportedFileError(
            f"{path.name} has {count} pages. Files can have up to {max_pages} pages, so try "
            "splitting it."
        )
    raw: list[str] = []
    length = 0
    for pdf_page in reader.pages:
        raw.append(pdf_page.extract_text() or "")
        length += len(raw[-1])
        if max_chars is not None and length > max_chars:  # stop early on a huge file
            raise _too_much_text(path.name, max_chars)
    offset = detect_page_offset(raw)
    texts = remove_boilerplate(raw) if len(raw) >= BOILERPLATE_MIN_PAGES else raw
    pages = []
    for number, text in enumerate(texts, start=1):
        text = clean_text(text)
        printed = number - offset if offset else None
        label = str(printed) if printed is not None and printed >= 1 else None
        if text:
            pages.append(Page(path.name, number, text, label))
    unreadable = [number for number, text in enumerate(raw, start=1) if not text.strip()]
    return DocumentText(pages, unreadable)


def check_archive(path: Path) -> None:
    """Refuse a zip archive (.pptx) that would expand to too much data, like a zip bomb.

    The sizes come from the archive's own directory. They can't be used to slip more data
    past these checks: Python's zipfile stops reading a member at its declared size.
    """
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
    total = sum(m.file_size for m in members)
    compressed = sum(m.compress_size for m in members)
    too_big = (
        len(members) > MAX_ARCHIVE_MEMBERS
        or total > MAX_ARCHIVE_BYTES
        or any(m.file_size > MAX_ARCHIVE_MEMBER_BYTES for m in members)
        or total > MAX_COMPRESSION_RATIO * max(compressed, 1)
        or any(
            m.file_size > max(MAX_COMPRESSION_RATIO * m.compress_size, 1024 * 1024) for m in members
        )
    )
    if too_big:
        raise UnsupportedFileError(
            f"{path.name} can't be read: its contents are much larger than a normal "
            "PowerPoint file."
        )


def load_pptx(path: Path) -> list[Page]:
    from pptx import Presentation

    check_archive(path)
    pages = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
        text = clean_text("\n".join(parts))
        if text:
            pages.append(Page(path.name, number, text))
    return pages


def load_text(path: Path) -> list[Page]:
    """Split Markdown/plain text on top-level headings so citations point at a section."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"\n(?=#{1,2} )", raw)
    pages = []
    for number, section in enumerate(sections, start=1):
        text = clean_text(section)
        if text:
            pages.append(Page(path.name, number, text))
    return pages


def read_document(
    path: Path, *, max_pages: int | None = None, max_chars: int | None = None
) -> DocumentText:
    """Read a document's pages. Pass limits when the file comes from someone else: a PDF with
    more than `max_pages` pages, or any file with more than `max_chars` characters of text,
    raises UnsupportedFileError instead."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path, max_pages, max_chars)
    if suffix == ".pptx":
        pages = load_pptx(path)
    elif suffix in (".md", ".txt"):
        pages = load_text(path)
    else:
        raise UnsupportedFileError(
            f"{path.name}: unsupported file type (supported: {', '.join(SUPPORTED_SUFFIXES)})"
        )
    _check_length(path.name, pages, max_chars)
    return DocumentText(pages)


def load_document(path: Path) -> list[Page]:
    return read_document(path).pages


def find_documents(paths: list[Path]) -> list[Path]:
    """Expand directories into the supported files they contain, sorted for determinism."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        else:
            found.append(path)
    return found
