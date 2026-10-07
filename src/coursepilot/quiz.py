"""Generate multiple-choice practice quizzes grounded in specific passages.

Each question must name the passage it was written from. Questions that fail validation
(wrong number of choices, duplicate choices, answer index out of range, or a source the
model wasn't given) are dropped rather than shown to a student.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from coursepilot.llm import LLMProvider
from coursepilot.retrieval import Hit, Mode, Retriever

SYSTEM_PROMPT = """You write practice quiz questions for a university course, using ONLY the \
course-material excerpts provided.

Each question must:
- test understanding of a concept in the excerpts, not trivia like page numbers or \
names of examples;
- have exactly 4 answer choices, one clearly correct, with plausible wrong choices;
- include a one or two sentence explanation of why the correct choice is right;
- list the label(s) of the excerpt(s) it is based on in `sources`, e.g. ["S2"]."""


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 answer choices")
    answer_index: int = Field(description="0-based index of the correct choice")
    explanation: str
    sources: list[str] = Field(description='Excerpt labels the question is based on, e.g. ["S1"]')


class QuizDraft(BaseModel):
    questions: list[QuizQuestion]


@dataclass
class Quiz:
    topic: str
    questions: list[QuizQuestion]
    passages: list[Hit]
    dropped: int

    def sources_for(self, question: QuizQuestion) -> list[Hit]:
        numbers = [int(s.lstrip("Ss")) for s in question.sources]
        return [self.passages[n - 1] for n in numbers]


def build_prompt(topic: str, hits: list[Hit], n: int) -> str:
    sources = "\n\n".join(
        f"[S{i}] ({hit.chunk.citation})\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )
    return f"Course-material excerpts:\n\n{sources}\n\nWrite {n} questions about: {topic}"


def validate_question(question: QuizQuestion, num_sources: int) -> bool:
    choices = [c.strip().lower() for c in question.choices]
    if len(choices) != 4 or len(set(choices)) != 4 or not all(choices):
        return False
    if not 0 <= question.answer_index < 4:
        return False
    if not question.sources:
        return False
    for label in question.sources:
        number = label.strip().lstrip("Ss")
        if not number.isdigit() or not 1 <= int(number) <= num_sources:
            return False
    return bool(question.question.strip() and question.explanation.strip())


def generate_quiz(
    retriever: Retriever, llm: LLMProvider, topic: str, n: int = 5, k: int = 8
) -> Quiz:
    mode = Mode.HYBRID_RERANK if retriever.reranker is not None else Mode.HYBRID
    hits = retriever.search(topic, k=k, mode=mode)
    draft = llm.generate(SYSTEM_PROMPT, build_prompt(topic, hits, n), QuizDraft)
    valid = [q for q in draft.questions if validate_question(q, len(hits))]
    return Quiz(topic, valid[:n], hits, dropped=len(draft.questions) - len(valid))
