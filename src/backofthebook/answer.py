"""Answer questions in a conversation, grounded in course materials when they cover it.

For each question the engine checks whether the materials (optionally just the documents the
student picked) contain anything close to it:
- if they do, the closest passages are added to the student's message, labeled [S1]..[Sn],
  and the model must cite them. After generation every citation is checked against the
  passages actually provided; citations to sources that don't exist are reported, not shown;
- if they don't, the model answers from general knowledge and must say so up front, rather
  than refusing or pretending the materials cover it.

Either way the model may reason, calculate, and solve problems step by step, and it sees the
earlier turns of the conversation so follow-up questions work.
"""

from __future__ import annotations

import re
import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field

from backofthebook.llm import LLMProvider, Message, StreamEvent
from backofthebook.retrieval import DEFAULT_MIN_SIMILARITY, Hit, Mode, Retriever

SYSTEM_PROMPT = """You are a friendly, sharp tutor helping a student study.

How to answer:
- Solve problems properly: show your reasoning, work through calculations step by step, and \
give a clear final answer. Use Markdown, including lists and LaTeX math ($...$) where helpful.
- When the student's message includes course-material excerpts, base your answer on them and \
cite the excerpt behind each claim in square brackets, exactly like [S1] or [S2][S3]. You may \
add your own explanation or worked steps, but don't contradict the excerpts.
- When the message says no relevant excerpts were found, start your answer with: \
"This isn't covered in your materials, so here's a general answer." Then answer from general \
knowledge, and don't use [S1]-style citations.
- Never invent sources. Be concise, clear, and encouraging, like a great TA."""

NOT_FOUND_MESSAGE = (
    "I couldn't find this in your course materials. Turn on an AI model in About > Settings "
    "to get answers to questions your materials don't cover."
)
NO_EXCERPTS_NOTE = "(No relevant excerpts were found in the student's course materials.)"

# Models don't always follow the requested format, so accept [S1], (S1), [S1][S2],
# (S1, S2), and [S1; S3]. Each bracketed group may list several labels.
_CITATION_GROUP = re.compile(r"[\[(]\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*[\])]")


def build_prompt(question: str, hits: list[Hit]) -> str:
    if not hits:
        return f"{NO_EXCERPTS_NOTE}\n\nStudent question: {question}"
    sources = "\n\n".join(
        f"[S{i}] ({hit.chunk.citation})\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )
    return f"Course-material excerpts:\n\n{sources}\n\nStudent question: {question}"


def extract_citations(text: str, num_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) source numbers cited in `text`, each in first-seen order."""
    valid: list[int] = []
    invalid: list[int] = []
    for group in _CITATION_GROUP.finditer(text):
        for number in re.findall(r"\d+", group.group()):
            n = int(number)
            bucket = valid if 1 <= n <= num_sources else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid


@dataclass
class Turn:
    """One finished exchange, kept so follow-up questions have context."""

    question: str
    answer: str


@dataclass
class Answer:
    question: str
    text: str = ""
    thinking: str = ""
    thinking_seconds: float = 0.0
    sources: list[Hit] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    grounded: bool = True  # False when answered from general knowledge

    @property
    def cited_hits(self) -> list[tuple[int, Hit]]:
        return [(n, self.sources[n - 1]) for n in self.cited]


def retrieval_query(question: str, history: list[Turn]) -> str:
    """Short follow-ups ("why?", "what about the median?") are searched together with the
    previous question so retrieval has something to go on."""
    if history and len(question.split()) < 6:
        return f"{history[-1].question} {question}"
    return question


class AnswerEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        k: int = 6,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        mode: Mode = Mode.HYBRID_RERANK,
        max_history: int = 6,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_similarity = min_similarity
        self.mode = mode if retriever.reranker is not None else Mode.HYBRID
        self.max_history = max_history

    def is_in_scope(self, question: str, sources: Collection[str] | None = None) -> bool:
        """Whether the materials contain anything semantically close to the question."""
        return self.retriever.top_similarity(question, sources) >= self.min_similarity

    def stream(
        self,
        question: str,
        history: list[Turn] | None = None,
        sources: Collection[str] | None = None,
    ) -> tuple[Answer, Iterator[StreamEvent]]:
        """Return the answer and its event stream; the answer is complete once the stream ends.

        `history` holds earlier turns of the conversation; `sources` limits retrieval to the
        given documents (e.g. the ones the student clicked).
        """
        history = (history or [])[-self.max_history :]
        answer = Answer(question)
        query = retrieval_query(question, history)
        if self.is_in_scope(query, sources):
            answer.sources = self.retriever.search(query, k=self.k, mode=self.mode, sources=sources)
        answer.grounded = bool(answer.sources)

        if not answer.grounded and self.llm.name == "extractive":
            answer.text = NOT_FOUND_MESSAGE
            return answer, iter([StreamEvent("text", NOT_FOUND_MESSAGE)])

        messages: list[Message] = []
        for turn in history:
            messages += [
                {"role": "user", "content": turn.question},
                {"role": "assistant", "content": turn.answer},
            ]
        messages.append({"role": "user", "content": build_prompt(question, answer.sources)})

        def events() -> Iterator[StreamEvent]:
            started = time.monotonic()
            thinking: list[str] = []
            text: list[str] = []
            for event in self.llm.chat_stream(SYSTEM_PROMPT, messages):
                if event.kind == "thinking":
                    thinking.append(event.text)
                else:
                    if not text:
                        answer.thinking_seconds = time.monotonic() - started
                    text.append(event.text)
                yield event
            answer.thinking = "".join(thinking)
            answer.text = "".join(text)
            if answer.grounded:
                answer.cited, answer.invalid_citations = extract_citations(
                    answer.text, len(answer.sources)
                )

        return answer, events()

    def ask(
        self,
        question: str,
        history: list[Turn] | None = None,
        sources: Collection[str] | None = None,
    ) -> Answer:
        answer, events = self.stream(question, history, sources)
        for _ in events:
            pass
        return answer
