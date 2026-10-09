import time
import zipfile
from pathlib import Path

import pytest
from conftest import make_pdf, make_pptx

from backofthebook import documents
from backofthebook.documents import (
    UnsupportedFileError,
    clean_text,
    detect_page_offset,
    find_documents,
    load_document,
    read_document,
    remove_boilerplate,
)


def test_clean_text_rejoins_hyphenated_words_and_collapses_whitespace():
    assert clean_text("data-\nset  is   peer -reviewed\n\n  ok") == "dataset is peer-reviewed\nok"


def test_clean_text_keeps_minus_signs_before_numbers():
    assert clean_text("x -1") == "x -1"


def test_clean_text_collapses_unicode_spaces_around_line_breaks():
    assert clean_text("a\u00a0\u2003 \n\u00a0\r\n  b\u00a0\u00a0c") == "a\nb c"


def seconds(action) -> float:
    start = time.perf_counter()
    action()
    return time.perf_counter() - start


@pytest.mark.parametrize("space", ["\u00a0", " \u00a0", "\u2003\t"])
def test_clean_text_is_fast_on_long_whitespace_runs(space: str):
    """A crafted page made of one long whitespace run used to take quadratic time."""
    text = "a" + space * (200_000 // len(space)) + "b"
    assert seconds(lambda: clean_text(text)) < 0.5


def test_running_head_detection_is_fast_on_long_lines():
    lines = ["\u2022" + "x  " * 70_000, "\u2022 " * 100_000 + "x"]
    raw = ["\n".join(lines)] * 3
    assert seconds(lambda: (detect_page_offset(raw), remove_boilerplate(raw))) < 0.5


def test_pdf_keeps_page_numbers(tmp_path: Path):
    pdf = make_pdf(
        tmp_path / "notes.pdf", ["Page one about variance.", "Page two about the median."]
    )

    pages = load_document(pdf)

    assert [(p.source, p.page) for p in pages] == [("notes.pdf", 1), ("notes.pdf", 2)]
    assert "median" in pages[1].text


def test_pdf_skips_blank_pages(tmp_path: Path):
    pdf = make_pdf(tmp_path / "gap.pdf", ["First page text.", "", "Third page text."])

    assert [p.page for p in load_document(pdf)] == [1, 3]


def test_detect_page_offset_from_running_heads():
    raw = [f"Body text {n}" for n in range(1, 11)]
    # Printed numbers are PDF page - 4, in alternating header/footer styles.
    for pdf_page in range(5, 11):
        printed = pdf_page - 4
        head = f"{printed}     2 • Chapter" if printed % 2 == 0 else f"2.1 • Section     {printed}"
        raw[pdf_page - 1] = f"{head}\n{raw[pdf_page - 1]}"

    assert detect_page_offset(raw) == 4


def test_detect_page_offset_needs_agreement():
    raw = ["10     1 • Chapter\ntext"] + ["plain text"] * 20
    assert detect_page_offset(raw) is None


def test_remove_boilerplate_drops_repeated_lines_and_running_heads():
    raw = [f"Access for free at openstax.org\nUnique content {n}" for n in range(10)]
    raw[0] = "10     1 • Chapter\n" + raw[0]

    cleaned = remove_boilerplate(raw)

    assert cleaned[0] == "Unique content 0"
    assert all("Access for free" not in page for page in cleaned)


def test_pptx_reads_slides_tables_and_notes(tmp_path: Path):
    deck = make_pptx(
        tmp_path / "lecture.pptx",
        [
            {"title": "Clustering", "body": "Group similar points"},
            {
                "title": "Metrics",
                "body": "Precision and recall",
                "table": [["Metric", "Formula"], ["Recall", "TP/(TP+FN)"]],
                "notes": "Remind students about F1.",
            },
        ],
    )

    pages = load_document(deck)

    assert [p.page for p in pages] == [1, 2]
    assert "Group similar points" in pages[0].text
    assert "Recall | TP/(TP+FN)" in pages[1].text
    assert "Remind students about F1." in pages[1].text


def test_markdown_splits_on_headings(tmp_path: Path):
    md = tmp_path / "notes.md"
    md.write_text("# Intro\nWhat data science is.\n\n## Methods\nRegression and trees.\n")

    pages = load_document(md)

    assert [p.page for p in pages] == [1, 2]
    assert pages[1].text.startswith("## Methods")


def test_unsupported_file_type_raises(tmp_path: Path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(UnsupportedFileError, match="unsupported"):
        load_document(path)


def test_find_documents_expands_folders(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "sub" / "a.txt").write_text("a")
    (tmp_path / "ignore.png").write_bytes(b"")

    found = find_documents([tmp_path])

    assert [p.name for p in found] == ["b.md", "a.txt"]


# --- Limits for untrusted uploads -----------------------------------------------------------


def test_pdf_over_the_page_limit_is_refused_before_reading_text(tmp_path: Path):
    pdf = make_pdf(tmp_path / "long.pdf", [f"Page {n} text." for n in range(1, 6)])

    with pytest.raises(UnsupportedFileError, match="5 pages"):
        read_document(pdf, max_pages=4)
    assert len(read_document(pdf, max_pages=5).pages) == 5


@pytest.mark.parametrize("suffix", [".pdf", ".md", ".pptx"])
def test_file_with_too_much_text_is_refused(tmp_path: Path, suffix: str):
    words = "variance " * 200
    if suffix == ".pdf":
        path = make_pdf(tmp_path / "big.pdf", [words[:90]] * 4)
    elif suffix == ".pptx":
        path = make_pptx(tmp_path / "big.pptx", [{"title": "T", "body": words}])
    else:
        path = tmp_path / "big.md"
        path.write_text(words)

    with pytest.raises(UnsupportedFileError, match="too much text"):
        read_document(path, max_chars=200)


def test_limits_are_off_by_default_for_the_cli(tmp_path: Path):
    pdf = make_pdf(tmp_path / "book.pdf", ["Some text."] * 3)
    assert len(read_document(pdf).pages) == 3


def test_pdf_is_parsed_once_and_reports_its_unreadable_pages(tmp_path: Path, monkeypatch):
    import pypdf

    opened = []

    class CountingReader(pypdf.PdfReader):
        def __init__(self, *args, **kwargs) -> None:
            opened.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pypdf, "PdfReader", CountingReader)
    pdf = make_pdf(tmp_path / "scan.pdf", ["Readable first page text here.", "", "Third page."])

    result = read_document(pdf)

    assert [p.page for p in result.pages] == [1, 3]
    assert result.unreadable == [2]
    assert len(opened) == 1


def zip_with(path: Path, members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_pptx_with_too_many_members_is_refused(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(documents, "MAX_ARCHIVE_MEMBERS", 10)
    deck = zip_with(tmp_path / "many.pptx", {f"ppt/f{n}.xml": b"<x/>" for n in range(11)})
    with pytest.raises(UnsupportedFileError, match="many"):
        load_document(deck)


def test_pptx_that_would_expand_too_much_is_refused(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(documents, "MAX_ARCHIVE_BYTES", 5_000)
    stored = zipfile.ZIP_STORED  # no compression, so only the size limits apply
    big_total = zip_with(tmp_path / "a.pptx", {"a": b"x" * 3_000, "b": b"x" * 3_000}, stored)
    with pytest.raises(UnsupportedFileError):
        load_document(big_total)

    monkeypatch.setattr(documents, "MAX_ARCHIVE_MEMBER_BYTES", 1_000)
    big_member = zip_with(tmp_path / "b.pptx", {"a": b"x" * 1_500}, stored)
    with pytest.raises(UnsupportedFileError):
        load_document(big_member)


def test_pptx_with_a_zip_bomb_member_is_refused(tmp_path: Path):
    bomb = zip_with(tmp_path / "bomb.pptx", {"ppt/slides/slide1.xml": b"\0" * (4 << 20)})
    with pytest.raises(UnsupportedFileError, match="bomb"):
        load_document(bomb)
