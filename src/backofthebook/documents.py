"""Load course materials into pages, keeping the page or slide number of every piece of text.

Citations are only as good as the location data behind them, so every loader returns
`Page` objects tagged with the file name and the 1-indexed page/slide they came from.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = (".pdf", ".pptx", ".md", ".txt")
# Repeated headers/footers are only stripped from long documents (books). In short documents a
# line on every page is usually real content, like the course name or the teacher's name.
BOILERPLATE_MIN_PAGES = 12


@dataclass(frozen=True)
class Page:
    source: str  # file name shown in citations, e.g. "lecture3.pdf"
    page: int  # 1-indexed page (PDF), slide (PPTX), or section (MD/TXT) number
    text: str
    label: str | None = None  # printed page number, when it differs from `page`


class UnsupportedFileError(ValueError):
    pass


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


def load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    raw = [pdf_page.extract_text() or "" for pdf_page in PdfReader(path).pages]
    offset = detect_page_offset(raw)
    texts = remove_boilerplate(raw) if len(raw) >= BOILERPLATE_MIN_PAGES else raw
    pages = []
    for number, text in enumerate(texts, start=1):
        text = clean_text(text)
        printed = number - offset if offset else None
        label = str(printed) if printed is not None and printed >= 1 else None
        if text:
            pages.append(Page(path.name, number, text, label))
    return pages


def load_pptx(path: Path) -> list[Page]:
    from pptx import Presentation

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


def load_document(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".pptx":
        return load_pptx(path)
    if suffix in (".md", ".txt"):
        return load_text(path)
    raise UnsupportedFileError(
        f"{path.name}: unsupported file type (supported: {', '.join(SUPPORTED_SUFFIXES)})"
    )


def unreadable_pages(path: Path) -> list[int]:
    """PDF pages with no extractable text, usually scanned images (empty for other types)."""
    if path.suffix.lower() != ".pdf":
        return []
    from pypdf import PdfReader

    return [
        number
        for number, page in enumerate(PdfReader(path).pages, start=1)
        if not (page.extract_text() or "").strip()
    ]


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
