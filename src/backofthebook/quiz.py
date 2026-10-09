"""Multiple-choice quizzes at three difficulty levels, always written from a source.

- **Course quiz** (`generate_quiz`): passages retrieved from your course materials.
- **Quiz generator** (`generate_anyquiz`): any topic; passages from the best-matching
  Wikipedia article.

Both modes work the same way after that: the model writes questions using ONLY the passages
and names the passage each question comes from. Then a separate source check shows each
question, its choices (not the answer key), and its cited passages to the model again, and
asks which choice the passages support. Questions whose answer key doesn't match, or that
the passages don't clearly support, are dropped. So are malformed and repeated questions.

Every failure raises a `BackOfTheBookError` with an error code (see `errors.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from backofthebook.answer import EXCERPT_RULE, excerpt
from backofthebook.errors import BackOfTheBookError, ErrorCode
from backofthebook.llm import LLMProvider
from backofthebook.retrieval import DEFAULT_MIN_SIMILARITY, Mode, Retriever
from backofthebook.wiki import Passage, WikipediaSource, article_passages

MAX_TOPIC_CHARS = 200
MIN_QUESTIONS, MAX_QUESTIONS = 1, 10
SPARE_QUESTIONS = 2  # extra questions requested so some can be dropped by the source check


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DIFFICULTY_GUIDE = {
    Difficulty.EASY: (
        "EASY: ask about the most well-known, central facts of the topic, the kind a casual fan "
        "or a student after one lesson would know. Use plain wording. The four choices must be "
        "clearly different from each other, and the wrong ones obviously wrong to someone who "
        "knows the basics. No obscure details, exact figures, or near-identical choices."
    ),
    Difficulty.MEDIUM: (
        "MEDIUM: test understanding, not just recall: why something happens, how two things "
        "compare, or applying an idea to a simple example. Wrong choices should be plausible "
        "but still clearly distinct."
    ),
    Difficulty.HARD: (
        "HARD: questions only someone who knows the topic well can answer: multi-step "
        "reasoning, specific lesser-known details, edge cases, or telling closely related ideas "
        "apart. Never ask basic definitions. Wrong choices should reflect common misconceptions."
    ),
}

WRITE_SYSTEM_PROMPT = f"""You write multiple-choice quiz questions using ONLY the source \
excerpts provided. Do not use outside knowledge. {EXCERPT_RULE}

Each question must:
- be answerable from the excerpts alone, with the correct answer clearly stated in them;
- test something meaningful about the topic, not trivia about the excerpts themselves \
(like page numbers, section names, or citation details);
- have exactly 4 answer choices with exactly one correct answer;
- give `answer_index` as the 0-based position of the correct choice;
- include a one or two sentence explanation of why the correct choice is right;
- list the id(s) of the excerpt(s) that state the answer in `sources`, e.g. ["S2"];
- read like a normal quiz: never mention "the excerpts", "the passage", "the text", \
"the source", or the [S1] labels in the question, the choices, or the explanation.

If the excerpts don't support enough good questions, write fewer. If the topic is harmful or \
not something a quiz should be written about, return an empty `questions` list."""

CHECK_SYSTEM_PROMPT = f"""You check quiz questions against their sources. For each question, \
read ONLY the excerpts shown with it and give the 0-based index of the choice those excerpts \
clearly support. If the excerpts don't clearly support exactly one choice, answer -1. Answer \
every question, in order. {EXCERPT_RULE}"""


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 answer choices")
    answer_index: int = Field(description="0-based index of the correct choice")
    explanation: str
    sources: list[str] = Field(description='Excerpt labels that state the answer, e.g. ["S1"]')


class QuizDraft(BaseModel):
    questions: list[QuizQuestion]


class AnswerSheet(BaseModel):
    answers: list[int] = Field(
        description="Per question, the 0-based index of the supported choice, or -1"
    )


@dataclass
class Quiz:
    topic: str
    difficulty: Difficulty
    questions: list[QuizQuestion]
    passages: list[Passage]
    source_title: str  # e.g. "your course materials" or "Super Mario Galaxy"
    source_url: str | None = None
    dropped: int = 0  # malformed or repeated questions removed
    failed_check: int = 0  # questions removed because the source check disagreed
    checked: bool = False  # True if the source check ran

    def sources_for(self, question: QuizQuestion) -> list[Passage]:
        return [self.passages[n - 1] for n in _source_numbers(question)]


class TopicNotCovered(BackOfTheBookError):
    def __init__(self, topic: str) -> None:
        super().__init__(
            ErrorCode.TOPIC_NOT_COVERED,
            f'Your course materials don\'t cover "{topic}". Try the Quiz generator to get a quiz '
            "on any topic from Wikipedia, or pick a topic from your notes or slides.",
            details={"topic": topic},
        )


def check_request(topic: str, n: int, difficulty: Difficulty | str) -> tuple[str, Difficulty]:
    """Validate and normalize a quiz request, raising a coded error for bad input."""
    topic = " ".join(topic.split())
    if not topic:
        raise BackOfTheBookError(ErrorCode.EMPTY_TOPIC, "Enter a topic to be quizzed on.")
    if len(topic) > MAX_TOPIC_CHARS:
        raise BackOfTheBookError(
            ErrorCode.TOPIC_TOO_LONG,
            f"Keep the topic under {MAX_TOPIC_CHARS} characters.",
            details={"length": len(topic)},
        )
    if not MIN_QUESTIONS <= n <= MAX_QUESTIONS:
        raise BackOfTheBookError(
            ErrorCode.INVALID_QUESTION_COUNT,
            f"Ask for between {MIN_QUESTIONS} and {MAX_QUESTIONS} questions.",
            details={"requested": n},
        )
    try:
        level = Difficulty(str(difficulty).lower())
    except ValueError:
        raise BackOfTheBookError(
            ErrorCode.INVALID_DIFFICULTY,
            "Difficulty must be easy, medium, or hard.",
            details={"difficulty": str(difficulty)},
        ) from None
    return topic, level


def _label_numbers(label: str) -> list[int]:
    """Excerpt numbers in a source label. Models write "S1", "[S1]", "s1", or "S1, S2"."""
    return [int(n) for n in re.findall(r"\d+", label)]


def _source_numbers(question: QuizQuestion) -> list[int]:
    numbers = (n for label in question.sources for n in _label_numbers(label))
    return list(dict.fromkeys(numbers))  # each excerpt once, in order


def validate_question(question: QuizQuestion, num_sources: int) -> bool:
    choices = [c.strip().lower() for c in question.choices]
    if len(choices) != 4 or len(set(choices)) != 4 or not all(choices):
        return False
    if not 0 <= question.answer_index < 4:
        return False
    if not question.sources:
        return False
    for label in question.sources:
        numbers = _label_numbers(label)
        if not numbers or not all(1 <= n <= num_sources for n in numbers):
            return False
    return bool(question.question.strip() and question.explanation.strip())


def _keep_valid(questions: list[QuizQuestion], num_sources: int) -> list[QuizQuestion]:
    """Drop malformed questions and repeats of an earlier question."""
    seen: set[str] = set()
    kept = []
    for q in questions:
        key = " ".join(q.question.lower().split())
        if key not in seen and validate_question(q, num_sources):
            seen.add(key)
            kept.append(q)
    return kept


def build_write_prompt(topic: str, passages: list[Passage], n: int, difficulty: Difficulty) -> str:
    excerpts = "\n\n".join(
        excerpt(f"S{i}", p.citation, p.text) for i, p in enumerate(passages, start=1)
    )
    return (
        f"Source excerpts:\n\n{excerpts}\n\n"
        f"Difficulty: {DIFFICULTY_GUIDE[difficulty]}\n\n"
        f"Write {n} questions about: {topic}"
    )


def build_check_prompt(questions: list[QuizQuestion], passages: list[Passage]) -> str:
    """Each question with its choices and cited excerpts, without the answer key."""
    blocks = []
    for i, q in enumerate(questions, start=1):
        choices = "\n".join(f"  {j}. {choice}" for j, choice in enumerate(q.choices))
        excerpts = "\n".join(
            excerpt(f"S{n}", passages[n - 1].citation, passages[n - 1].text)
            for n in _source_numbers(q)
        )
        blocks.append(f"Question {i}: {q.question}\n{choices}\nExcerpts:\n{excerpts}")
    return "\n\n".join(blocks)


def source_check(
    llm: LLMProvider, questions: list[QuizQuestion], passages: list[Passage]
) -> tuple[list[QuizQuestion], bool]:
    """Keep questions whose answer key matches what their cited passages support.

    Returns (kept questions, whether the check ran). If the answer sheet doesn't line up with
    the questions, the check is asked for once more, then skipped rather than guessing which
    answer belongs to which.
    """
    prompt = build_check_prompt(questions, passages)
    for _ in range(2):
        sheet = llm.generate(CHECK_SYSTEM_PROMPT, prompt, AnswerSheet)
        if len(sheet.answers) == len(questions):
            pairs = zip(questions, sheet.answers, strict=True)
            return [q for q, answer in pairs if answer == q.answer_index], True
    return questions, False


def _quiz_from_passages(
    llm: LLMProvider,
    topic: str,
    level: Difficulty,
    n: int,
    passages: list[Passage],
    source_title: str,
    source_url: str | None,
    check: bool,
) -> Quiz:
    requested = n + SPARE_QUESTIONS if check else n
    draft = llm.generate(
        WRITE_SYSTEM_PROMPT, build_write_prompt(topic, passages, requested, level), QuizDraft
    )
    valid = _keep_valid(draft.questions, len(passages))
    if not valid:
        raise BackOfTheBookError(
            ErrorCode.NO_VALID_QUESTIONS,
            f'Couldn\'t write usable questions about "{topic}" from {source_title}. '
            "Try a more specific topic, or try again.",
            details={"topic": topic},
        )
    dropped = len(draft.questions) - len(valid)
    checked, failed = False, 0
    if check:
        kept, checked = source_check(llm, valid, passages)
        failed = len(valid) - len(kept)
        if not kept:
            raise BackOfTheBookError(
                ErrorCode.NO_VALID_QUESTIONS,
                f'None of the questions about "{topic}" passed the source check, so none were '
                "shown. Try again or try a more specific topic.",
                details={"topic": topic, "failed_check": failed},
            )
        valid = kept
    return Quiz(
        topic,
        level,
        valid[:n],
        passages,
        source_title,
        source_url,
        dropped=dropped,
        failed_check=failed,
        checked=checked,
    )


def generate_quiz(
    retriever: Retriever,
    llm: LLMProvider,
    topic: str,
    n: int = 5,
    difficulty: Difficulty | str = Difficulty.MEDIUM,
    check: bool = True,
    k: int = 8,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Quiz:
    """Course quiz: questions from passages in the loaded course materials."""
    topic, level = check_request(topic, n, difficulty)
    if retriever.top_similarity(topic) < min_similarity:
        raise TopicNotCovered(topic)
    mode = Mode.HYBRID_RERANK if retriever.reranker is not None else Mode.HYBRID
    passages = [
        Passage(hit.chunk.citation, hit.chunk.text, source=hit.chunk.source)
        for hit in retriever.search(topic, k=k, mode=mode)
    ]
    return _quiz_from_passages(llm, topic, level, n, passages, "your course materials", None, check)


def generate_anyquiz(
    llm: LLMProvider,
    topic: str,
    n: int = 5,
    difficulty: Difficulty | str = Difficulty.MEDIUM,
    check: bool = True,
    source: WikipediaSource | None = None,
) -> Quiz:
    """Quiz generator: any topic, with questions from the best-matching Wikipedia article."""
    topic, level = check_request(topic, n, difficulty)
    article = (source or WikipediaSource()).find_article(topic)
    passages = article_passages(article)
    return _quiz_from_passages(
        llm, topic, level, n, passages, f"Wikipedia: {article.title}", article.url, check
    )
