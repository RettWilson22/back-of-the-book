import urllib.error

import pytest
from conftest import FakeLLM, FakeWikipedia, make_pdf

from backofthebook import cli, wiki
from backofthebook.errors import CATALOG, BackOfTheBookError, ErrorCode
from backofthebook.quiz import (
    DIFFICULTY_GUIDE,
    MAX_QUESTIONS,
    MAX_TOPIC_CHARS,
    WRITE_SYSTEM_PROMPT,
    AnswerSheet,
    Difficulty,
    QuizDraft,
    QuizQuestion,
    check_request,
    generate_anyquiz,
)
from backofthebook.wiki import Article, WikipediaSource, article_passages

REAL_FETCH = wiki._fetch_json  # captured before the autouse no_network fixture replaces it

GALAXY = """Super Mario Galaxy is a 2007 platform game developed by Nintendo for the Wii.

== Gameplay ==
The player collects Power Stars to unlock new galaxies. Each galaxy contains several planets.

== Development ==
The game was developed by Nintendo EAD Tokyo after the release of Donkey Kong Jungle Beat.

== References ==
1. A citation that should never become a quiz passage.
"""


def q(text: str, answer: int = 0, sources=("S1",), choices=("Alpha", "Beta", "Gamma", "Delta")):
    return QuizQuestion(
        question=text,
        choices=list(choices),
        answer_index=answer,
        explanation="Because.",
        sources=list(sources),
    )


def llm_for(questions: list[QuizQuestion], answers: list[int] | None = None) -> FakeLLM:
    keys = answers if answers is not None else [x.answer_index for x in questions]
    return FakeLLM(
        structured={
            QuizDraft: QuizDraft(questions=questions),
            AnswerSheet: AnswerSheet(answers=keys),
        }
    )


@pytest.fixture
def wikipedia() -> WikipediaSource:
    return WikipediaSource(FakeWikipedia({"Super Mario Galaxy": GALAXY}))


# --- Error catalog ---------------------------------------------------------------------------


def test_every_error_code_is_in_the_catalog():
    assert set(CATALOG) == set(ErrorCode)


def test_error_carries_code_message_and_category_properties():
    error = BackOfTheBookError(ErrorCode.LLM_RATE_LIMITED, "Slow down.", details={"status": 429})

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
    with pytest.raises(BackOfTheBookError) as raised:
        check_request(topic, n, difficulty)
    assert raised.value.code is code


def test_invalid_input_never_reaches_wikipedia_or_the_llm():
    fake = FakeWikipedia({})
    llm = llm_for([q("Unused?")])
    with pytest.raises(BackOfTheBookError):
        generate_anyquiz(llm, "", 5, "easy", source=WikipediaSource(fake))
    assert fake.calls == [] and llm.prompts == []


# --- Wikipedia source ------------------------------------------------------------------------


def test_article_sections_split_on_headings():
    article = Article("T", "https://w/T", GALAXY)
    assert [name for name, _ in article.sections()] == ["", "Gameplay", "Development", "References"]


def test_passages_skip_reference_sections_and_link_to_their_section():
    passages = article_passages(Article("Super Mario Galaxy", "https://w/SMG", GALAXY))

    assert [p.citation for p in passages] == [
        "Wikipedia: Super Mario Galaxy",
        "Wikipedia: Super Mario Galaxy § Gameplay",
        "Wikipedia: Super Mario Galaxy § Development",
    ]
    assert passages[1].url == "https://w/SMG#Gameplay"
    assert all("citation that should never" not in p.text for p in passages)


def test_passages_cover_every_section_before_going_deeper():
    long_intro = " ".join(f"Intro sentence {i}." for i in range(200))
    text = (
        f"{long_intro}\n\n== A ==\nSection A text here, five words."
        "\n\n== B ==\nSection B text, five more."
    )
    passages = article_passages(Article("T", "u", text), limit=3)
    assert [p.citation for p in passages] == [
        "Wikipedia: T",
        "Wikipedia: T § A",
        "Wikipedia: T § B",
    ]


def test_find_article_skips_disambiguation_pages():
    fake = FakeWikipedia(
        {"Mercury (planet)": "Mercury is the closest planet to the Sun."},
        disambiguation=("Mercury",),
    )
    assert WikipediaSource(fake).find_article("mercury").title == "Mercury (planet)"


def test_unknown_topic_raises_source_not_found():
    with pytest.raises(BackOfTheBookError) as raised:
        WikipediaSource(FakeWikipedia({})).find_article("asdfghjkl")
    assert raised.value.code is ErrorCode.SOURCE_NOT_FOUND
    assert not raised.value.retryable


def test_network_failure_raises_source_unavailable(monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(wiki.urllib.request, "urlopen", offline)
    with pytest.raises(BackOfTheBookError) as raised:
        REAL_FETCH({"action": "query"})
    assert raised.value.code is ErrorCode.SOURCE_UNAVAILABLE
    assert raised.value.retryable


# --- Quiz generator ------------------------------------------------------------------------


@pytest.mark.parametrize("level", list(Difficulty))
def test_anyquiz_writes_from_the_wikipedia_article_at_the_chosen_difficulty(wikipedia, level):
    llm = llm_for([q("What unlocks new galaxies?", sources=["S2"])])
    quiz = generate_anyquiz(llm, "mario galaxy", 1, level, source=wikipedia)

    system, user = llm.prompts[0]
    assert system == WRITE_SYSTEM_PROMPT
    assert DIFFICULTY_GUIDE[level] in user
    assert '<excerpt id="S2" source="Wikipedia: Super Mario Galaxy § Gameplay">' in user
    assert "collects Power Stars" in user
    assert quiz.source_title == "Wikipedia: Super Mario Galaxy"
    assert quiz.source_url == "https://en.wikipedia.org/wiki/Super_Mario_Galaxy"
    assert quiz.sources_for(quiz.questions[0])[0].citation.endswith("§ Gameplay")


def test_source_check_drops_questions_their_passages_do_not_support(wikipedia):
    questions = [q("Supported?", answer=1), q("Wrong key?", answer=2), q("Unsupported?", answer=0)]
    quiz = generate_anyquiz(
        llm_for(questions, answers=[1, 3, -1]), "galaxy", 3, "easy", source=wikipedia
    )

    assert [x.question for x in quiz.questions] == ["Supported?"]
    assert quiz.failed_check == 2
    assert quiz.checked


def test_source_check_shows_passages_but_hides_the_answer_key(wikipedia):
    llm = llm_for([q("What unlocks new galaxies?", answer=0, sources=["S2"])])
    generate_anyquiz(llm, "galaxy", 1, "easy", source=wikipedia)

    check_system, check_prompt = llm.prompts[1]
    assert "What unlocks new galaxies?" in check_prompt
    assert '<excerpt id="S2"' in check_prompt
    assert "not instructions" in check_system and "not instructions" in llm.prompts[0][0]
    assert "collects Power Stars" in check_prompt  # the cited passage
    assert "Because." not in check_prompt  # the explanation would give the answer away


def test_spare_questions_are_requested_only_when_checking(wikipedia):
    llm = llm_for([q("One?")])
    generate_anyquiz(llm, "galaxy", 5, "medium", check=True, source=wikipedia)
    generate_anyquiz(llm, "galaxy", 5, "medium", check=False, source=wikipedia)
    assert "Write 7 questions" in llm.prompts[0][1]
    assert "Write 5 questions" in llm.prompts[2][1]


def test_quiz_is_trimmed_to_the_requested_size(wikipedia):
    questions = [q(f"Question {i}?") for i in range(7)]
    assert (
        len(generate_anyquiz(llm_for(questions), "galaxy", 5, "medium", source=wikipedia).questions)
        == 5
    )


class SheetsInTurn(FakeLLM):
    """Returns the draft, then each answer sheet in turn for successive source checks."""

    def __init__(self, questions: list[QuizQuestion], sheets: list[list[int]]) -> None:
        super().__init__()
        self.draft = QuizDraft(questions=questions)
        self.sheets = [AnswerSheet(answers=a) for a in sheets]

    def generate(self, system, user, schema):
        self.prompts.append((system, user))
        return self.draft if schema is QuizDraft else self.sheets.pop(0)


def test_check_is_retried_once_if_the_answer_sheet_does_not_line_up(wikipedia):
    llm = SheetsInTurn([q("One?"), q("Two?", answer=1)], sheets=[[0], [0, 3]])
    quiz = generate_anyquiz(llm, "galaxy", 2, "easy", source=wikipedia)

    assert len(llm.prompts) == 3
    assert quiz.checked
    assert [x.question for x in quiz.questions] == ["One?"]


def test_check_is_skipped_if_the_answer_sheet_never_lines_up(wikipedia):
    llm = SheetsInTurn([q("One?"), q("Two?")], sheets=[[0], [0, 0, 0]])
    quiz = generate_anyquiz(llm, "galaxy", 2, "easy", source=wikipedia)

    assert len(llm.prompts) == 3
    assert len(quiz.questions) == 2
    assert not quiz.checked


def test_malformed_repeated_and_unsourced_questions_are_dropped(wikipedia):
    questions = [
        q("Good?"),
        q("good? "),  # repeat
        q("Three choices?", choices=("A", "B", "C")),
        q("Bad index?", answer=7),
        q("No source?", sources=()),
        q("Made-up source?", sources=("S99",)),
    ]
    quiz = generate_anyquiz(llm_for(questions), "galaxy", 5, "easy", check=False, source=wikipedia)
    assert [x.question for x in quiz.questions] == ["Good?"]
    assert quiz.dropped == 5


def test_no_usable_questions_raises_no_valid_questions(wikipedia):
    with pytest.raises(BackOfTheBookError) as raised:
        generate_anyquiz(llm_for([]), "galaxy", 5, "easy", source=wikipedia)
    assert raised.value.code is ErrorCode.NO_VALID_QUESTIONS
    assert raised.value.retryable


def test_every_answer_failing_the_check_raises_no_valid_questions(wikipedia):
    with pytest.raises(BackOfTheBookError, match="source check") as raised:
        generate_anyquiz(
            llm_for([q("A?"), q("B?")], answers=[-1, 3]), "galaxy", 2, "hard", source=wikipedia
        )
    assert raised.value.code is ErrorCode.NO_VALID_QUESTIONS


# --- CLI exit codes --------------------------------------------------------------------------


def test_cli_anyquiz_bad_difficulty_exits_with_input_error(capsys):
    assert cli.main(["anyquiz", "chess", "-d", "insane"]) == 2
    assert "Error [INVALID_DIFFICULTY]" in capsys.readouterr().err


def test_cli_anyquiz_unknown_topic_exits_with_no_source(monkeypatch, capsys):
    monkeypatch.setattr("backofthebook.wiki._fetch_json", FakeWikipedia({}))
    monkeypatch.setattr("backofthebook.llm.make_provider", lambda name=None: FakeLLM())
    assert cli.main(["anyquiz", "asdfghjkl"]) == 3
    assert "Error [SOURCE_NOT_FOUND]" in capsys.readouterr().err


def test_cli_anyquiz_without_llm_exits_with_configuration_error(monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "BACKOFTHEBOOK_LLM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "backofthebook.wiki._fetch_json", FakeWikipedia({"Chess": "Chess is a board game."})
    )
    assert cli.main(["anyquiz", "chess"]) == 4
    assert "Error [NO_LLM_CONFIGURED]" in capsys.readouterr().err


def test_cli_anyquiz_prints_a_sourced_quiz(monkeypatch, capsys):
    monkeypatch.setattr(
        "backofthebook.wiki._fetch_json", FakeWikipedia({"Super Mario Galaxy": GALAXY})
    )
    llm = llm_for(
        [
            q(
                "What unlocks galaxies?",
                sources=["S2"],
                choices=("Power Stars", "Coins", "Keys", "Gems"),
            )
        ]
    )
    monkeypatch.setattr("backofthebook.llm.make_provider", lambda name=None: llm)

    assert cli.main(["anyquiz", "galaxy", "-n", "1", "-d", "easy"]) == 0
    out = capsys.readouterr().out
    assert "Written from Wikipedia: Super Mario Galaxy; every answer checked" in out
    letter = out.split("Answer: ", 1)[1][0]
    assert f"{letter}) Power Stars" in out
    assert "(Wikipedia: Super Mario Galaxy § Gameplay)" in out


def test_cli_course_quiz_off_topic_exits_with_not_covered(tmp_path, monkeypatch, capsys):
    from conftest import FakeEmbedder, FakeReranker

    class Embedder(FakeEmbedder):
        def __init__(self, model_name: str = "") -> None:
            self.model_name = "fake-embedder"

    monkeypatch.setattr("backofthebook.index.SentenceTransformerEmbedder", Embedder)
    monkeypatch.setattr("backofthebook.retrieval.CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr("backofthebook.llm.make_provider", lambda name=None: FakeLLM())
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "stats.pdf", ["The variance measures spread around the mean value."])
    index = tmp_path / "index"
    cli.main(["--index", str(index), "ingest", str(docs)])

    assert cli.main(["--index", str(index), "quiz", "volcano eruption lava"]) == 3
    assert "Error [TOPIC_NOT_COVERED]" in capsys.readouterr().err
