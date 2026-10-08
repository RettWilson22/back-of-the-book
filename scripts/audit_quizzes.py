"""Generate quizzes for a fixed set of topics, for grading answer-key accuracy by hand.

Compares two ways of writing an AnyQuiz quiz on the same topics:
- memory:  the model writes questions from its own knowledge (the original AnyQuiz design,
           kept here only as a baseline);
- sourced: questions from the topic's Wikipedia article, then checked against the cited
           passages (what AnyQuiz does now).

Writes one JSON line per question to eval/anyquiz/questions.jsonl, plus each topic's Wikipedia
article text to eval/anyquiz/articles/ (git-ignored) so every answer key can be graded against
the same source. Grades go in eval/anyquiz/grades.jsonl; `--report` prints the accuracy table.

Usage:
    GROQ_API_KEY=... python scripts/audit_quizzes.py           # generate
    python scripts/audit_quizzes.py --report                   # summarize grades
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from backofthebook.errors import BackOfTheBookError
from backofthebook.llm import LLMProvider, make_provider
from backofthebook.quiz import (
    DIFFICULTY_GUIDE,
    Difficulty,
    QuizDraft,
    generate_anyquiz,
    validate_question,
)
from backofthebook.wiki import WikipediaSource

OUT = Path(__file__).resolve().parents[1] / "eval" / "anyquiz"
TOPICS = [
    "Super Mario Galaxy",
    "French Revolution",
    "Photosynthesis",
    "Black hole",
    "The Great Gatsby",
    "Python (programming language)",
    "Mitochondrion",
    "World War I",
    "Basketball",
    "Pokémon Red and Blue",
]
N, LEVEL = 5, Difficulty.MEDIUM

MEMORY_PROMPT = """You write multiple-choice quiz questions on the topic the user asks about, \
from your own knowledge. Each question has exactly 4 choices with one correct answer, \
`answer_index` is its 0-based position, include a short explanation, and leave `sources` as \
["S1"]. Only ask about well-established facts."""


T = TypeVar("T")


def with_retries(action: Callable[[], T], attempts: int = 6) -> T:
    """Retry rate limits and other retryable errors (the free Groq tier is rate limited)."""
    for attempt in range(attempts):
        try:
            return action()
        except BackOfTheBookError as e:
            if not e.retryable or attempt == attempts - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"    {e.code}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def memory_quiz(llm: LLMProvider, topic: str) -> list[dict[str, Any]]:
    prompt = f"Topic: {topic}\nDifficulty: {DIFFICULTY_GUIDE[LEVEL]}\nWrite {N} questions."
    draft = with_retries(lambda: llm.generate(MEMORY_PROMPT, prompt, QuizDraft))
    return [
        {
            "question": q.question,
            "choices": q.choices,
            "key": q.answer_index,
            "explanation": q.explanation,
            "sources": [],
        }
        for q in draft.questions
        if validate_question(q, 1)
    ][:N]


def sourced_quiz(llm: LLMProvider, topic: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quiz = with_retries(lambda: generate_anyquiz(llm, topic, N, LEVEL))
    rows = [
        {
            "question": q.question,
            "choices": q.choices,
            "key": q.answer_index,
            "explanation": q.explanation,
            "sources": [{"citation": p.citation, "text": p.text} for p in quiz.sources_for(q)],
        }
        for q in quiz.questions
    ]
    stats = {"dropped": quiz.dropped, "failed_check": quiz.failed_check, "checked": quiz.checked}
    return rows, stats


def generate() -> None:
    llm = make_provider()
    if llm.name == "extractive":
        sys.exit("Set GROQ_API_KEY (or ANTHROPIC_API_KEY) first.")
    (OUT / "articles").mkdir(parents=True, exist_ok=True)
    wikipedia = WikipediaSource()
    rows = []
    for topic in TOPICS:
        print(f"{topic}...", file=sys.stderr)
        article = wikipedia.find_article(topic)
        slug = article.title.replace(" ", "_").replace("/", "-")
        (OUT / "articles" / f"{slug}.txt").write_text(f"{article.url}\n\n{article.text}")
        for i, row in enumerate(memory_quiz(llm, topic)):
            rows.append({"id": f"{slug}/memory/{i}", "topic": topic, "mode": "memory", **row})
        sourced, stats = sourced_quiz(llm, topic)
        print(f"    sourced: {stats}", file=sys.stderr)
        for i, row in enumerate(sourced):
            rows.append({"id": f"{slug}/sourced/{i}", "topic": topic, "mode": "sourced", **row})
    out = OUT / "questions.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"Wrote {len(rows)} questions to {out}", file=sys.stderr)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percent(count: int, total: int) -> str:
    return f"{count} ({count / total:.0%})" if total else "0"


def report() -> None:
    questions = {row["id"]: row for row in read_jsonl(OUT / "questions.jsonl")}
    by_mode: dict[str, Counter[str]] = {}
    for grade in read_jsonl(OUT / "grades.jsonl"):
        mode = questions[grade["id"]]["mode"]
        by_mode.setdefault(mode, Counter())[grade["grade"]] += 1
    print("| Approach | Questions | Correct key | Wrong key | Ambiguous |")
    print("|---|---|---|---|---|")
    for mode in ("memory", "sourced"):
        counts = by_mode.get(mode, Counter())
        total = sum(counts.values())
        cells = " | ".join(percent(counts[k], total) for k in ("correct", "wrong", "ambiguous"))
        print(f"| {mode} | {total} | {cells} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="summarize eval/anyquiz/grades.jsonl")
    report() if parser.parse_args().report else generate()
