"""Drive the Streamlit app headlessly with fake models and a fake LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import TOY_PAGES, FakeEmbedder, FakeLLM, FakeReranker
from streamlit.testing.v1 import AppTest

from coursepilot.chunking import chunk_pages
from coursepilot.index import CorpusIndex
from coursepilot.quiz import QuizDraft, QuizQuestion

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def start(llm: FakeLLM) -> AppTest:
        index_dir = tmp_path / "index"
        CorpusIndex.build(chunk_pages(TOY_PAGES), FakeEmbedder()).save(index_dir)
        monkeypatch.setenv("COURSEPILOT_INDEX", str(index_dir))
        monkeypatch.setenv("COURSEPILOT_SAMPLE", "0")

        class Embedder(FakeEmbedder):
            def __init__(self, model_name: str = "") -> None:
                self.model_name = "fake-embedder"

        monkeypatch.setattr("coursepilot.index.SentenceTransformerEmbedder", Embedder)
        monkeypatch.setattr("coursepilot.retrieval.CrossEncoderReranker", FakeReranker)
        monkeypatch.setattr("coursepilot.llm.make_provider", lambda name=None: llm)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception, at.exception
        return at

    return start


def test_app_shows_loaded_materials(app):
    at = app(FakeLLM())
    assert any("2 file(s)" in c.value for c in at.sidebar.caption)


def test_ask_streams_answer_and_shows_cited_sources(app):
    at = app(FakeLLM("A decision tree asks yes or no questions [S1]."))
    at.chat_input[0].set_value("how does a decision tree split data with questions").run()

    assert not at.exception
    assert any("yes or no questions [S1]" in m.value for m in at.markdown)
    assert at.expander[0].label.startswith("[S1] ml.pptx, p. 1")


def test_off_topic_question_is_declined(app):
    at = app(FakeLLM("should not appear"))
    at.chat_input[0].set_value("volcano eruption lava").run()
    assert any("couldn't find this" in m.value for m in at.markdown)


def test_quiz_generates_and_grades_answers(app):
    draft = QuizDraft(
        questions=[
            QuizQuestion(
                question="What does k-means assign points to?",
                choices=["Nearest centroid", "Random cluster", "Largest cluster", "First point"],
                answer_index=0,
                explanation="Each point joins the closest centroid.",
                sources=["S1"],
            )
        ]
    )
    at = app(FakeLLM(structured=draft))
    at.text_input[0].set_value("k-means clustering")
    at.button[0].click().run()  # the form's "Generate quiz" button

    assert at.radio[0].options == draft.questions[0].choices
    at.radio[0].set_value("Nearest centroid")
    next(b for b in at.button if b.label == "Check answers").click().run()

    assert not at.exception
    assert any(s.value == "Score: 1 / 1" for s in at.subheader)
