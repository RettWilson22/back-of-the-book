"""The AI should read a student's own documents fully, not just a few search hits.

Regression tests for a real report: "I asked who the teacher was and it said the document
didn't say, but it's on the front page."
"""

from pathlib import Path

from conftest import TOY_PAGES, FakeEmbedder, FakeLLM, FakeReranker, make_pdf

from backofthebook.answer import AnswerEngine
from backofthebook.chunking import chunk_pages
from backofthebook.documents import load_document, unreadable_pages
from backofthebook.index import CorpusIndex
from backofthebook.retrieval import Retriever

HEADER = "COMP 4970 Mobile App Development - Instructor: Dr. Maria Rivera"


def syllabus(tmp_path: Path) -> Path:
    return make_pdf(
        tmp_path / "syllabus.pdf",
        [
            f"{HEADER}\nOffice hours are Tuesdays from two to four in Shelby Center room 2101.",
            f"{HEADER}\nGrading is forty percent projects, thirty percent exams, and quizzes.",
            f"{HEADER}\nThe final project is due on the last day of class before noon.",
        ],
    )


def engine_with(tmp_path: Path, llm: FakeLLM, **kwargs) -> AnswerEngine:
    pages = TOY_PAGES + load_document(syllabus(tmp_path))
    index = CorpusIndex.build(chunk_pages(pages), FakeEmbedder())
    return AnswerEngine(Retriever(index, FakeEmbedder(), FakeReranker()), llm, **kwargs)


def test_headers_on_every_page_of_a_short_document_are_kept(tmp_path):
    pages = load_document(syllabus(tmp_path))
    assert all("Dr. Maria Rivera" in p.text for p in pages)


def test_short_chosen_document_is_read_in_full(tmp_path):
    llm = FakeLLM("Your teacher is Dr. Maria Rivera [S1].")
    engine = engine_with(tmp_path, llm)

    answer = engine.ask("who is the teacher?", sources=["syllabus.pdf"])

    prompt = llm.prompts[0][1]
    assert "Dr. Maria Rivera" in prompt
    assert "Office hours" in prompt and "final project" in prompt  # the whole document
    assert "decision tree" not in prompt  # nothing from documents it wasn't asked about
    assert answer.grounded and answer.cited == [1]


def test_long_documents_still_include_their_first_page(tmp_path):
    llm = FakeLLM("Answer [S1].")
    engine = engine_with(tmp_path, llm, full_text_words=10, min_similarity=0.0)

    engine.ask("when is the final project due", sources=["syllabus.pdf"])

    assert "Office hours" in llm.prompts[0][1]  # page 1 is always included
    assert "final project" in llm.prompts[0][1]


def test_scanned_pages_are_reported(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", ["Readable first page text here.", ""])
    assert unreadable_pages(pdf) == [2]
    assert unreadable_pages(tmp_path / "notes.md") == []
