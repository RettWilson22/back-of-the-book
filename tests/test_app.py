"""Drive the Streamlit app headlessly with fake models and a fake LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import TOY_PAGES, FakeEmbedder, FakeLLM, FakeReranker, FakeWikipedia
from streamlit.testing.v1 import AppTest

from coursepilot.chunking import chunk_pages
from coursepilot.index import CorpusIndex
from coursepilot.quiz import AnswerSheet, QuizDraft, QuizQuestion

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


def kmeans_question(**overrides) -> QuizQuestion:
    fields = {
        "question": "What does k-means assign points to?",
        "choices": ["Nearest centroid", "Random cluster", "Largest cluster", "First point"],
        "answer_index": 0,
        "explanation": "Each point joins the closest centroid.",
        "sources": ["S1"],
    }
    return QuizQuestion(**{**fields, **overrides})


def question_radio(at: AppTest, text: str):
    return next(r for r in at.radio if text in r.label)


def quiz_llm(questions: list[QuizQuestion], answers: list[int] | None = None) -> FakeLLM:
    keys = answers if answers is not None else [x.answer_index for x in questions]
    return FakeLLM(
        structured={
            QuizDraft: QuizDraft(questions=questions),
            AnswerSheet: AnswerSheet(answers=keys),
        }
    )


def test_course_quiz_generates_checks_and_grades_answers(app):
    llm = quiz_llm([kmeans_question()])
    at = app(llm)
    at.text_input[0].set_value("k-means clustering")
    at.radio[0].set_value("Hard")
    at.button[0].click().run()  # "Generate quiz" in the course tab

    assert "HARD" in llm.prompts[0][1]
    radio = question_radio(at, "What does k-means assign points to?")
    assert radio.options == kmeans_question().choices
    radio.set_value("Nearest centroid")
    next(b for b in at.button if b.label == "Check answers").click().run()

    assert not at.exception
    assert any(s.value == "Score: 1 / 1" for s in at.subheader)
    assert any("checked against its source" in c.value for c in at.caption)
    assert any(
        "Source: ml.pptx, p. 2" in c.value and "Report a problem" in c.value for c in at.caption
    )


def test_course_quiz_on_off_topic_subject_points_to_anyquiz(app):
    at = app(FakeLLM(structured=QuizDraft(questions=[])))
    at.text_input[0].set_value("volcano eruption lava")
    at.button[0].click().run()

    assert not at.exception
    assert any("don't cover" in i.value and "AnyQuiz" in i.value for i in at.info)
    assert any("TOPIC_NOT_COVERED" in c.value for c in at.caption)


GALAXY = (
    "Super Mario Galaxy is a 2007 Wii game.\n\n"
    "== Gameplay ==\nThe player collects Power Stars to unlock new galaxies."
)


def test_anyquiz_writes_from_wikipedia_checks_and_grades(app, monkeypatch):
    monkeypatch.setattr(
        "coursepilot.wiki._fetch_json", FakeWikipedia({"Super Mario Galaxy": GALAXY})
    )
    star = kmeans_question(
        question="What do you collect in Super Mario Galaxy to unlock new galaxies?",
        choices=["Power Stars", "Coins", "Mushrooms", "Keys"],
        explanation="Power Stars open up new galaxies.",
        sources=["S2"],
    )
    wrong_key = kmeans_question(question="A question with a bad answer key?", sources=["S1"])
    at = app(quiz_llm([star, wrong_key], answers=[0, 2]))
    at.text_input[1].set_value("mario galaxy")
    at.button[1].click().run()  # "Generate quiz" in the AnyQuiz tab

    assert not at.exception
    assert not any("bad answer key" in r.label for r in at.radio)  # dropped by the source check
    assert any("Written from [Wikipedia: Super Mario Galaxy]" in c.value for c in at.caption)
    question_radio(at, "Super Mario Galaxy").set_value("Power Stars")
    next(b for b in at.button if b.label == "Check answers").click().run()
    assert any(s.value == "Score: 1 / 1" for s in at.subheader)
    source = next(c.value for c in at.caption if c.value.startswith("Source:"))
    assert "Super_Mario_Galaxy#Gameplay" in source
    assert "github.com/RettWilson22/coursepilot/issues/new" in source


def test_anyquiz_unknown_topic_shows_source_not_found(app, monkeypatch):
    monkeypatch.setattr("coursepilot.wiki._fetch_json", FakeWikipedia({}))
    at = app(quiz_llm([]))
    at.text_input[1].set_value("asdfghjkl")
    at.button[1].click().run()

    assert not at.exception
    assert any("Couldn't find a Wikipedia article" in i.value for i in at.info)
    assert any("SOURCE_NOT_FOUND" in c.value for c in at.caption)


def test_anyquiz_shows_retryable_error_code_when_nothing_usable_comes_back(app, monkeypatch):
    monkeypatch.setattr(
        "coursepilot.wiki._fetch_json", FakeWikipedia({"Super Mario Galaxy": GALAXY})
    )
    at = app(quiz_llm([]))
    at.text_input[1].set_value("mario galaxy")
    at.button[1].click().run()

    assert not at.exception
    assert any("Trying again may work" in e.value for e in at.error)
    assert any("NO_VALID_QUESTIONS" in c.value for c in at.caption)


def test_unexpected_errors_are_reported_as_internal_errors(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("coursepilot.quiz.generate_anyquiz", boom)
    at = app(FakeLLM())
    at.text_input[1].set_value("photosynthesis")
    at.button[1].click().run()

    assert not at.exception
    assert any("INTERNAL_ERROR" in c.value for c in at.caption)
