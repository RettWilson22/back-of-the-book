import pytest
from conftest import FakeLLM, make_pdf

from coursepilot import cli
from coursepilot.errors import CATALOG, CoursePilotError, ErrorCode
from coursepilot.quiz import (
    ANYQUIZ_SYSTEM_PROMPT,
    DIFFICULTY_GUIDE,
    MAX_QUESTIONS,
    MAX_TOPIC_CHARS,
    AnswerSheet,
    Difficulty,
    QuizDraft,
    QuizQuestion,
    check_request,
    generate_anyquiz,
)


def q(text: str, answer: int = 0, choices=("Alpha", "Beta", "Gamma", "Delta")) -> QuizQuestion:
    return QuizQuestion(
        question=text, choices=list(choices), answer_index=answer, explanation="Because."
    )


def llm_for(questions: list[QuizQuestion], answers: list[int] | None = None) -> FakeLLM:
    sheet = AnswerSheet(
        answers=answers if answers is not None else [x.answer_index for x in questions]
    )
    return FakeLLM(structured={QuizDraft: QuizDraft(questions=questions), AnswerSheet: sheet})


# --- Error catalog ---------------------------------------------------------------------------


def test_every_error_code_is_in_the_catalog():
    assert set(CATALOG) == set(ErrorCode)


def test_error_carries_code_message_and_category_properties():
    error = CoursePilotError(ErrorCode.LLM_RATE_LIMITED, "Slow down.", details={"status": 429})

    assert str(error) == "Slow down."
    assert error.retryable
    assert error.exit_code == 5
    assert error.to_dict() == {
        "code": "LLM_RATE_LIMITED",
        "message": "Slow down.",
        "retryable": True,
        "details": {"status": 429},
    }


def test_input_errors_are_not_retryable_and_exit_with_2():
    for code in (ErrorCode.EMPTY_TOPIC, ErrorCode.TOPIC_TOO_LONG, ErrorCode.INVALID_DIFFICULTY):
        assert (CATALOG[code].retryable, CATALOG[code].exit_code) == (False, 2)


# --- Request validation ----------------------------------------------------------------------


def test_check_request_normalizes_topic_and_difficulty():
    assert check_request("  Super   Mario\nGalaxy ", 5, "HARD") == (
        "Super Mario Galaxy",
        Difficulty.HARD,
    )


@pytest.mark.parametrize(
    ("topic", "n", "difficulty", "code"),
    [
        ("   ", 5, "easy", ErrorCode.EMPTY_TOPIC),
        ("x" * (MAX_TOPIC_CHARS + 1), 5, "easy", ErrorCode.TOPIC_TOO_LONG),
        ("history", 0, "easy", ErrorCode.INVALID_QUESTION_COUNT),
        ("history", MAX_QUESTIONS + 1, "easy", ErrorCode.INVALID_QUESTION_COUNT),
        ("history", 5, "impossible", ErrorCode.INVALID_DIFFICULTY),
    ],
)
def test_check_request_rejects_bad_input(topic, n, difficulty, code):
    with pytest.raises(CoursePilotError) as raised:
        check_request(topic, n, difficulty)
    assert raised.value.code is code


def test_invalid_input_never_reaches_the_llm():
    llm = llm_for([q("Unused?")])
    with pytest.raises(CoursePilotError):
        generate_anyquiz(llm, "", 5, "easy")
    assert llm.prompts == []


# --- AnyQuiz generation ----------------------------------------------------------------------


@pytest.mark.parametrize("level", list(Difficulty))
def test_anyquiz_prompt_includes_the_difficulty_guide(level: Difficulty):
    llm = llm_for([q("What is the capital of France?")])
    generate_anyquiz(llm, "geography", 3, level)

    system, user = llm.prompts[0]
    assert system == ANYQUIZ_SYSTEM_PROMPT
    assert DIFFICULTY_GUIDE[level] in user
    assert "Topic: geography" in user


def test_anyquiz_trims_to_the_requested_number_of_questions():
    questions = [q(f"Question {i}?") for i in range(7)]
    quiz = generate_anyquiz(llm_for(questions), "trivia", 5, "medium")

    assert len(quiz.questions) == 5
    assert quiz.checked and quiz.failed_check == 0


def test_anyquiz_asks_for_spares_only_when_checking():
    llm = llm_for([q("Only one?")])
    generate_anyquiz(llm, "trivia", 5, "medium", verify=True)
    generate_anyquiz(llm, "trivia", 5, "medium", verify=False)
    assert "Write 7 questions." in llm.prompts[0][1]
    assert "Write 5 questions." in llm.prompts[2][1]


def test_self_check_drops_questions_with_a_disputed_answer_key():
    questions = [q("Agreed?", answer=1), q("Disputed?", answer=2), q("Also agreed?", answer=0)]
    quiz = generate_anyquiz(llm_for(questions, answers=[1, 3, 0]), "trivia", 3, "easy")

    assert [x.question for x in quiz.questions] == ["Agreed?", "Also agreed?"]
    assert quiz.failed_check == 1
    assert quiz.checked
    assert not quiz.grounded


def test_self_check_hides_the_answer_key_from_the_checker():
    llm = llm_for([q("Which is first?", answer=0)])
    generate_anyquiz(llm, "trivia", 1, "easy")
    _, verify_prompt = llm.prompts[1]
    assert "Which is first?" in verify_prompt
    assert "Because." not in verify_prompt and "answer_index" not in verify_prompt


def test_self_check_is_skipped_if_the_answer_sheet_does_not_line_up():
    questions = [q("One?"), q("Two?")]
    quiz = generate_anyquiz(llm_for(questions, answers=[0]), "trivia", 2, "easy")
    assert len(quiz.questions) == 2
    assert not quiz.checked


def test_anyquiz_drops_malformed_and_repeated_questions():
    questions = [
        q("Good?"),
        q("good? "),  # repeat
        q("Three choices?", choices=("A", "B", "C")),
        q("Bad index?", answer=7),
    ]
    quiz = generate_anyquiz(llm_for(questions), "trivia", 5, "easy", verify=False)
    assert [x.question for x in quiz.questions] == ["Good?"]
    assert quiz.dropped == 3


def test_empty_quiz_raises_no_valid_questions():
    with pytest.raises(CoursePilotError) as raised:
        generate_anyquiz(llm_for([]), "something harmful", 5, "easy")
    assert raised.value.code is ErrorCode.NO_VALID_QUESTIONS
    assert raised.value.retryable


def test_quiz_where_every_answer_is_disputed_raises_no_valid_questions():
    with pytest.raises(CoursePilotError, match="self-check") as raised:
        generate_anyquiz(llm_for([q("A?"), q("B?")], answers=[1, 1]), "obscure topic", 2, "hard")
    assert raised.value.code is ErrorCode.NO_VALID_QUESTIONS


# --- CLI exit codes --------------------------------------------------------------------------


def test_cli_anyquiz_bad_difficulty_exits_with_input_error(capsys):
    assert cli.main(["anyquiz", "chess", "-d", "insane"]) == 2
    assert "Error [INVALID_DIFFICULTY]" in capsys.readouterr().err


def test_cli_anyquiz_without_llm_exits_with_configuration_error(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "COURSEPILOT_LLM"):
        monkeypatch.delenv(var, raising=False)
    assert cli.main(["anyquiz", "chess"]) == 4
    assert "Error [NO_LLM_CONFIGURED]" in capsys.readouterr().err


def test_cli_anyquiz_prints_a_checked_quiz(monkeypatch, capsys):
    llm = llm_for(
        [q("Which piece moves in an L shape?", choices=("Knight", "Rook", "Bishop", "Pawn"))]
    )
    monkeypatch.setattr("coursepilot.llm.make_provider", lambda name=None: llm)

    assert cli.main(["anyquiz", "chess", "-n", "1", "-d", "easy"]) == 0
    out = capsys.readouterr().out
    assert "chess (easy)" in out
    assert "double-checked" in out
    assert "A) Knight" in out and "Answer: A." in out


def test_cli_course_quiz_off_topic_exits_with_not_covered(tmp_path, monkeypatch, capsys):
    from conftest import FakeEmbedder, FakeReranker

    class Embedder(FakeEmbedder):
        def __init__(self, model_name: str = "") -> None:
            self.model_name = "fake-embedder"

    monkeypatch.setattr("coursepilot.index.SentenceTransformerEmbedder", Embedder)
    monkeypatch.setattr("coursepilot.retrieval.CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr("coursepilot.llm.make_provider", lambda name=None: FakeLLM())
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "stats.pdf", ["The variance measures spread around the mean value."])
    index = tmp_path / "index"
    cli.main(["--index", str(index), "ingest", str(docs)])

    assert cli.main(["--index", str(index), "quiz", "volcano eruption lava"]) == 3
    assert "Error [TOPIC_NOT_COVERED]" in capsys.readouterr().err
