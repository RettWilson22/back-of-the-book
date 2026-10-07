"""Multiple-choice quizzes at three difficulty levels, in two modes.

- **Course quiz** (`generate_quiz`): questions written ONLY from retrieved course passages.
  Each question names the passage it came from. Topics outside the materials are declined.
- **AnyQuiz** (`generate_anyquiz`): questions on any topic, from the model's own knowledge.
  Because nothing grounds those answers, an optional self-check has the model answer its own
  questions without the answer key, and questions where the two disagree are dropped.

In both modes, malformed questions (not exactly 4 distinct choices, an out-of-range answer,
blank text, duplicates, or a source the model wasn't given) are dropped rather than shown.
Every failure raises a `CoursePilotError` with an error code (see `errors.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

from coursepilot.errors import CoursePilotError, ErrorCode
from coursepilot.llm import LLMProvider
from coursepilot.retrieval import DEFAULT_MIN_SIMILARITY, Hit, Mode, Retriever

MAX_TOPIC_CHARS = 200
MIN_QUESTIONS, MAX_QUESTIONS = 1, 10


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DIFFICULTY_GUIDE = {
    Difficulty.EASY: (
        "EASY: test recall of basic facts, terms, and definitions. Use plain wording. "
        "Wrong choices should be clearly wrong to someone who knows the basics."
    ),
    Difficulty.MEDIUM: (
        "MEDIUM: test understanding and application: explaining why, comparing ideas, or "
        "applying a concept to a simple example. Wrong choices should be plausible."
    ),
    Difficulty.HARD: (
        "HARD: test deep understanding: multi-step reasoning, edge cases, or distinguishing "
        "closely related ideas. Wrong choices should reflect common misconceptions."
    ),
}

_QUESTION_RULES = """Each question must:
- have exactly 4 answer choices with exactly one correct answer;
- give `answer_index` as the 0-based position of the correct choice;
- include a one or two sentence explanation of why the correct choice is right."""

COURSE_SYSTEM_PROMPT = f"""You write practice quiz questions for a university course, using \
ONLY the course-material excerpts provided.

{_QUESTION_RULES}
- test a concept in the excerpts, not trivia like page numbers or names of examples;
- list the label(s) of the excerpt(s) it is based on in `sources`, e.g. ["S2"]."""

ANYQUIZ_SYSTEM_PROMPT = f"""You write multiple-choice quiz questions on whatever topic the user \
asks about, from your own knowledge.

{_QUESTION_RULES}
- only ask about facts that are well established and that you are confident about. If the \
topic is too obscure, ambiguous, or recent to write accurate questions, write fewer questions \
rather than guessing;
- avoid questions whose answer depends on opinion or on events that may have changed;
- leave `sources` as an empty list.

If the topic is harmful or not something a quiz should be written about, return an empty \
`questions` list."""

VERIFY_SYSTEM_PROMPT = """Answer each multiple-choice question independently. For each one, \
give the 0-based index of the correct choice, in the same order as the questions."""


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 answer choices")
    answer_index: int = Field(description="0-based index of the correct choice")
    explanation: str
    sources: list[str] = Field(
        default_factory=list, description='Excerpt labels the question is based on, e.g. ["S1"]'
    )


class QuizDraft(BaseModel):
    questions: list[QuizQuestion]


class AnswerSheet(BaseModel):
    answers: list[int] = Field(description="0-based index of the chosen answer, one per question")


@dataclass
class Quiz:
    topic: str
    difficulty: Difficulty
    questions: list[QuizQuestion]
    passages: list[Hit] = field(default_factory=list)  # empty for AnyQuiz
    dropped: int = 0  # malformed or duplicate questions removed
    failed_check: int = 0  # AnyQuiz questions removed because the self-check disagreed
    checked: bool = False  # True if the AnyQuiz self-check ran successfully

    @property
    def grounded(self) -> bool:
        return bool(self.passages)

    def sources_for(self, question: QuizQuestion) -> list[Hit]:
        numbers = [int(s.strip().lstrip("Ss")) for s in question.sources]
        return [self.passages[n - 1] for n in numbers]


class TopicNotCovered(CoursePilotError):
    def __init__(self, topic: str) -> None:
        super().__init__(
            ErrorCode.TOPIC_NOT_COVERED,
            f'Your course materials don\'t cover "{topic}". Try AnyQuiz to get a quiz on any '
            "topic, or pick a topic from your notes or slides.",
            details={"topic": topic},
        )


def check_request(topic: str, n: int, difficulty: Difficulty | str) -> tuple[str, Difficulty]:
    """Validate and normalize a quiz request, raising a coded error for bad input."""
    topic = " ".join(topic.split())
    if not topic:
        raise CoursePilotError(ErrorCode.EMPTY_TOPIC, "Enter a topic to be quizzed on.")
    if len(topic) > MAX_TOPIC_CHARS:
        raise CoursePilotError(
            ErrorCode.TOPIC_TOO_LONG,
            f"Keep the topic under {MAX_TOPIC_CHARS} characters.",
            details={"length": len(topic)},
        )
    if not MIN_QUESTIONS <= n <= MAX_QUESTIONS:
        raise CoursePilotError(
            ErrorCode.INVALID_QUESTION_COUNT,
            f"Ask for between {MIN_QUESTIONS} and {MAX_QUESTIONS} questions.",
            details={"requested": n},
        )
    try:
        level = Difficulty(str(difficulty).lower())
    except ValueError:
        raise CoursePilotError(
            ErrorCode.INVALID_DIFFICULTY,
            "Difficulty must be easy, medium, or hard.",
            details={"difficulty": str(difficulty)},
        ) from None
    return topic, level


def validate_question(
    question: QuizQuestion, num_sources: int = 0, require_sources: bool = True
) -> bool:
    choices = [c.strip().lower() for c in question.choices]
    if len(choices) != 4 or len(set(choices)) != 4 or not all(choices):
        return False
    if not 0 <= question.answer_index < 4:
        return False
    if require_sources:
        if not question.sources:
            return False
        for label in question.sources:
            number = label.strip().lstrip("Ss")
            if not number.isdigit() or not 1 <= int(number) <= num_sources:
                return False
    return bool(question.question.strip() and question.explanation.strip())


def _keep_valid(
    questions: list[QuizQuestion], num_sources: int, require_sources: bool
) -> list[QuizQuestion]:
    """Drop malformed questions and repeats of an earlier question."""
    seen: set[str] = set()
    kept = []
    for q in questions:
        key = " ".join(q.question.lower().split())
        if key not in seen and validate_question(q, num_sources, require_sources):
            seen.add(key)
            kept.append(q)
    return kept


def _no_questions(topic: str, hint: str) -> CoursePilotError:
    return CoursePilotError(
        ErrorCode.NO_VALID_QUESTIONS,
        f'Couldn\'t write usable questions about "{topic}". {hint}',
        details={"topic": topic},
    )


def build_course_prompt(topic: str, hits: list[Hit], n: int, difficulty: Difficulty) -> str:
    sources = "\n\n".join(
        f"[S{i}] ({hit.chunk.citation})\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )
    return (
        f"Course-material excerpts:\n\n{sources}\n\n"
        f"Difficulty: {DIFFICULTY_GUIDE[difficulty]}\n\n"
        f"Write {n} questions about: {topic}"
    )


def generate_quiz(
    retriever: Retriever,
    llm: LLMProvider,
    topic: str,
    n: int = 5,
    difficulty: Difficulty | str = Difficulty.MEDIUM,
    k: int = 8,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Quiz:
    """Course quiz: questions grounded in the loaded materials."""
    topic, level = check_request(topic, n, difficulty)
    if retriever.top_similarity(topic) < min_similarity:
        raise TopicNotCovered(topic)
    mode = Mode.HYBRID_RERANK if retriever.reranker is not None else Mode.HYBRID
    hits = retriever.search(topic, k=k, mode=mode)
    draft = llm.generate(
        COURSE_SYSTEM_PROMPT, build_course_prompt(topic, hits, n, level), QuizDraft
    )
    valid = _keep_valid(draft.questions, len(hits), require_sources=True)
    if not valid:
        raise _no_questions(topic, "Try a broader topic, or try again.")
    return Quiz(topic, level, valid[:n], hits, dropped=len(draft.questions) - len(valid))


def build_anyquiz_prompt(topic: str, n: int, difficulty: Difficulty) -> str:
    return f"Topic: {topic}\nDifficulty: {DIFFICULTY_GUIDE[difficulty]}\nWrite {n} questions."


def build_verify_prompt(questions: list[QuizQuestion]) -> str:
    blocks = []
    for i, q in enumerate(questions, start=1):
        choices = "\n".join(f"  {j}. {choice}" for j, choice in enumerate(q.choices))
        blocks.append(f"Question {i}: {q.question}\n{choices}")
    return "\n\n".join(blocks)


def self_check(llm: LLMProvider, questions: list[QuizQuestion]) -> tuple[list[QuizQuestion], bool]:
    """Keep questions the model answers the same way without seeing the key.

    Returns (kept questions, whether the check ran). If the model's answer sheet doesn't line
    up with the questions, the check is skipped rather than guessing which answer is which.
    """
    sheet = llm.generate(VERIFY_SYSTEM_PROMPT, build_verify_prompt(questions), AnswerSheet)
    if len(sheet.answers) != len(questions):
        return questions, False
    kept = [q for q, a in zip(questions, sheet.answers, strict=True) if a == q.answer_index]
    return kept, True


def generate_anyquiz(
    llm: LLMProvider,
    topic: str,
    n: int = 5,
    difficulty: Difficulty | str = Difficulty.MEDIUM,
    verify: bool = True,
) -> Quiz:
    """AnyQuiz: questions on any topic from the model's own knowledge, optionally self-checked."""
    topic, level = check_request(topic, n, difficulty)
    # Ask for a couple of spares when checking, since some questions may be dropped.
    requested = min(n + 2, MAX_QUESTIONS + 2) if verify else n
    draft = llm.generate(
        ANYQUIZ_SYSTEM_PROMPT, build_anyquiz_prompt(topic, requested, level), QuizDraft
    )
    valid = _keep_valid(draft.questions, 0, require_sources=False)
    if not valid:
        raise _no_questions(
            topic, "It may be too obscure, too vague, or not a suitable quiz topic."
        )
    dropped = len(draft.questions) - len(valid)
    checked, failed = False, 0
    if verify:
        kept, checked = self_check(llm, valid)
        failed = len(valid) - len(kept)
        if not kept:
            raise _no_questions(
                topic,
                "The self-check couldn't confirm any of the answers, so none were shown. "
                "Try a more specific or better-known topic.",
            )
        valid = kept
    return Quiz(topic, level, valid[:n], dropped=dropped, failed_check=failed, checked=checked)
