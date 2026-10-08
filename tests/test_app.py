"""Drive the Streamlit app headlessly with fake models and a fake LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import TOY_PAGES, FakeEmbedder, FakeLLM, FakeReranker, FakeWikipedia
from streamlit.testing.v1 import AppTest

from backofthebook.chunking import chunk_pages
from backofthebook.index import CorpusIndex
from backofthebook.quiz import AnswerSheet, QuizDraft, QuizQuestion

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def start(llm: FakeLLM) -> AppTest:
        index_dir = tmp_path / "index"
        CorpusIndex.build(chunk_pages(TOY_PAGES), FakeEmbedder()).save(index_dir)
        monkeypatch.setenv("BACKOFTHEBOOK_INDEX", str(index_dir))
        monkeypatch.setenv("BACKOFTHEBOOK_SAMPLE", "0")

        class Embedder(FakeEmbedder):
            def __init__(self, model_name: str = "") -> None:
                self.model_name = "fake-embedder"

        monkeypatch.setattr("backofthebook.index.SentenceTransformerEmbedder", Embedder)
        monkeypatch.setattr("backofthebook.retrieval.CrossEncoderReranker", FakeReranker)
        monkeypatch.setattr("backofthebook.llm.make_provider", lambda name=None: llm)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception, at.exception
        return at

    return start


def quiz_buttons(at: AppTest) -> list[Any]:
    """Buttons other than the Ask tab's example questions, in page order."""
    return [b for b in at.button if not (b.key or "").startswith("example_")]


def test_app_shows_loaded_materials_as_clickable_documents(app):
    at = app(FakeLLM())
    assert at.file_uploader  # the upload box lives in the Ask tab, not a sidebar
    assert not at.sidebar.caption
    assert at.button_group[0].options == ["ml.pptx", "stats.pdf"]
    assert any("Answering from all loaded documents" in c.value for c in at.caption)


def test_clicking_a_document_answers_only_from_it(app):
    llm = FakeLLM("From the stats notes [S1].")
    at = app(llm)
    at.button_group[0].select("stats.pdf").run()
    assert any("Answering from stats.pdf" in c.value for c in at.caption)

    ask(at, "how is the mean computed from the sum of the values")
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert labels and all("stats.pdf" in label for label in labels)


def ask(at: AppTest, question: str) -> AppTest:
    return at.chat_input[0].set_value(question).run()


def test_ask_shows_thinking_answer_and_cited_sources(app):
    llm = FakeLLM("A decision tree asks yes or no questions [S1].", thinking="Use S1.")
    at = ask(app(llm), "how does a decision tree split data with questions")

    assert not at.exception
    assert any("yes or no questions [S1]" in m.value for m in at.markdown)
    assert [s.label for s in at.get("status")] == ["Thought for a moment"]
    assert any(e.label.startswith("[S1] ml.pptx, p. 1") for e in at.expander)


def test_follow_up_question_sends_the_whole_conversation(app):
    llm = FakeLLM("Splits on questions [S1].")
    at = ask(app(llm), "how does a decision tree split data with questions")
    ask(at, "why?")

    assert not at.exception
    roles = [m["role"] for m in llm.conversations[-1]]
    assert roles == ["user", "assistant", "user"]
    assert any(b.label == "New chat" for b in at.button)


def test_off_topic_question_is_answered_and_labeled_general_knowledge(app):
    llm = FakeLLM("This isn't covered in your materials, so here's a general answer.")
    at = ask(app(llm), "volcano eruption lava")

    assert not at.exception
    assert any("general answer" in m.value for m in at.markdown)
    assert any("general knowledge, not from your materials" in c.value for c in at.caption)


def test_thinking_text_is_escaped_not_rendered_as_html(app):
    llm = FakeLLM("Fine [S1].", thinking="<img src=x onerror=alert(1)>")
    at = ask(app(llm), "how does a decision tree split data with questions")
    rendered = " ".join(m.value for m in at.markdown)
    assert "&lt;img" in rendered and "<img" not in rendered


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
    quiz_buttons(at)[0].click().run()  # "Generate quiz" in the course tab

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
    quiz_buttons(at)[0].click().run()

    assert not at.exception
    assert any("don't cover" in i.value and "Quiz generator" in i.value for i in at.info)
    assert not any("TOPIC_NOT_COVERED" in c.value for c in at.caption)  # a hint, not a fault


GALAXY = (
    "Super Mario Galaxy is a 2007 Wii game.\n\n"
    "== Gameplay ==\nThe player collects Power Stars to unlock new galaxies."
)


def test_anyquiz_writes_from_wikipedia_checks_and_grades(app, monkeypatch):
    monkeypatch.setattr(
        "backofthebook.wiki._fetch_json", FakeWikipedia({"Super Mario Galaxy": GALAXY})
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
    quiz_buttons(at)[1].click().run()  # "Generate quiz" in the Quiz generator tab

    assert not at.exception
    assert not any("bad answer key" in r.label for r in at.radio)  # dropped by the source check
    assert any("Written from [Wikipedia: Super Mario Galaxy]" in c.value for c in at.caption)
    question_radio(at, "Super Mario Galaxy").set_value("Power Stars")
    next(b for b in at.button if b.label == "Check answers").click().run()
    assert any(s.value == "Score: 1 / 1" for s in at.subheader)
    source = next(c.value for c in at.caption if c.value.startswith("Source:"))
    assert "Super_Mario_Galaxy#Gameplay" in source
    assert "github.com/RettWilson22/back-of-the-book/issues/new" in source


def test_anyquiz_unknown_topic_shows_source_not_found(app, monkeypatch):
    monkeypatch.setattr("backofthebook.wiki._fetch_json", FakeWikipedia({}))
    at = app(quiz_llm([]))
    at.text_input[1].set_value("asdfghjkl")
    quiz_buttons(at)[1].click().run()

    assert not at.exception
    assert any("Couldn't find a Wikipedia article" in i.value for i in at.info)
    assert not any("SOURCE_NOT_FOUND" in c.value for c in at.caption)  # a hint, not a fault


def test_anyquiz_shows_retryable_error_code_when_nothing_usable_comes_back(app, monkeypatch):
    monkeypatch.setattr(
        "backofthebook.wiki._fetch_json", FakeWikipedia({"Super Mario Galaxy": GALAXY})
    )
    at = app(quiz_llm([]))
    at.text_input[1].set_value("mario galaxy")
    quiz_buttons(at)[1].click().run()

    assert not at.exception
    assert any("Trying again may work" in e.value for e in at.error)
    assert any("NO_VALID_QUESTIONS" in c.value for c in at.caption)


def test_unexpected_errors_are_reported_as_internal_errors(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("backofthebook.quiz.generate_anyquiz", boom)
    at = app(FakeLLM())
    at.text_input[1].set_value("photosynthesis")
    quiz_buttons(at)[1].click().run()

    assert not at.exception
    assert any("INTERNAL_ERROR" in c.value for c in at.caption)


def test_api_key_is_read_from_streamlit_secrets(app, tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.secrets["GROQ_API_KEY"] = "test-key-from-secrets"
    app(FakeLLM())  # sets up the fake index and models
    at.run()
    import os

    assert not at.exception
    assert os.environ.get("GROQ_API_KEY") == "test-key-from-secrets"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def uploaded(name: str, text: str):
    """Stand-in for a processed upload, as stored in the session."""
    from types import SimpleNamespace

    from backofthebook.documents import Page

    index = CorpusIndex.build(chunk_pages([Page(name, 1, text)]), FakeEmbedder())
    return SimpleNamespace(name=name, index=index, pages=1, notes=[])


def test_uploaded_documents_are_listed_used_by_default_and_removable(app):
    at = app(FakeLLM("Your teacher is Dr. Rivera [S1]."))
    at.session_state["docs"] = {
        "syllabus.pdf": uploaded(
            "syllabus.pdf", "Instructor: Dr. Maria Rivera, office hours Tuesdays."
        )
    }
    at.run()

    assert not at.exception
    assert any(m.value == "**Documents**" for m in at.markdown)
    assert any(b.key == "remove_syllabus.pdf" and b.label == "Remove" for b in at.button)
    assert "syllabus.pdf" in at.button_group[0].options
    assert any("Answering from syllabus.pdf" in c.value for c in at.caption)

    at.button_group[0].select("syllabus.pdf").run()
    next(b for b in at.button if b.key == "remove_syllabus.pdf").click().run()

    assert not at.exception
    assert at.session_state["docs"] == {}
    assert "syllabus.pdf" not in at.button_group[0].options
    assert any("Answering from all loaded documents" in c.value for c in at.caption)


def test_example_question_button_asks_it(app):
    llm = FakeLLM("A mean is the average [S1].")
    at = app(llm)
    example = next(b for b in at.button if (b.key or "").startswith("example_"))
    example.click().run()

    assert not at.exception
    assert example.label in llm.prompts[0][1]
    assert not any((b.key or "").startswith("example_") for b in at.button)  # hidden after


def test_wrong_answer_shows_what_the_student_chose(app):
    at = app(quiz_llm([kmeans_question()]))
    at.text_input[0].set_value("k-means clustering")
    quiz_buttons(at)[0].click().run()
    question_radio(at, "What does k-means").set_value("Random cluster")
    next(b for b in at.button if b.label == "Check answers").click().run()

    assert any(s.value == "Score: 0 / 1" for s in at.subheader)
    assert any("You chose *Random cluster*" in m.value for m in at.markdown)


def test_input_mistakes_show_a_hint_without_an_error_code(app):
    at = app(FakeLLM())
    quiz_buttons(at)[1].click().run()  # empty topic

    assert not at.exception
    assert at.info
    assert not any("EMPTY_TOPIC" in c.value for c in at.caption)


def test_unanswered_questions_are_marked_skipped(app):
    at = app(quiz_llm([kmeans_question()]))
    at.text_input[0].set_value("k-means clustering")
    quiz_buttons(at)[0].click().run()
    next(b for b in at.button if b.label == "Check answers").click().run()

    assert any("Skipped" in m.value for m in at.markdown)
    assert not any("You chose" in m.value for m in at.markdown)


def test_sample_can_be_hidden_while_there_are_uploads_and_brought_back(app):
    at = app(FakeLLM("Your teacher is Dr. Rivera [S1]."))
    at.session_state["docs"] = {
        "syllabus.pdf": uploaded(
            "syllabus.pdf", "Instructor: Dr. Maria Rivera, office hours Tuesdays."
        )
    }
    at.run()
    base_sources = [o for o in at.button_group[0].options if o != "syllabus.pdf"]
    assert base_sources

    next(b for b in at.button if b.key == "hide_sample").click().run()
    assert not at.exception
    assert at.button_group[0].options == ["syllabus.pdf"]

    next(b for b in at.button if b.key == "show_sample").click().run()
    assert not at.exception, at.exception
    assert set(base_sources) <= set(at.button_group[0].options)


def test_sample_comes_back_when_the_last_upload_is_removed(app):
    at = app(FakeLLM())
    at.session_state["docs"] = {
        "syllabus.pdf": uploaded(
            "syllabus.pdf", "Instructor: Dr. Maria Rivera, office hours Tuesdays."
        )
    }
    at.run()
    next(b for b in at.button if b.key == "hide_sample").click().run()
    next(b for b in at.button if b.key == "remove_syllabus.pdf").click().run()

    assert not at.exception
    assert at.button_group[0].options  # the built-in material is back, so Ask still works
    assert not any(b.key == "hide_sample" for b in at.button)  # nothing to fall back on
