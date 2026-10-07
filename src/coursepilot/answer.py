"""Answer questions from retrieved passages, with citations that are checked, not trusted.

The model sees passages labeled [S1]..[Sn] and must cite them. After generation, every
citation is validated against the passages actually provided; citations to sources that
don't exist are reported instead of being shown as real.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from coursepilot.llm import LLMProvider
from coursepilot.retrieval import DEFAULT_MIN_SIMILARITY, Hit, Mode, Retriever

SYSTEM_PROMPT = """You are a teaching assistant answering a student's question using ONLY the \
course-material excerpts provided.

- Base every statement on the excerpts. Do not add outside knowledge.
- Cite the excerpt(s) supporting each statement in square brackets, exactly like [S1] or \
[S2][S3].
- If the excerpts do not contain the answer, say that the course materials don't cover it \
and stop. Do not guess.
- Be concise and clear, the way a good TA explains things. Use short paragraphs or bullets."""

NOT_FOUND_MESSAGE = (
    "I couldn't find this in your course materials, so I won't guess. "
    "Try rephrasing, or check that the right documents are loaded."
)

# Models don't always follow the requested format, so accept [S1], (S1), [S1][S2],
# (S1, S2), and [S1; S3]. Each bracketed group may list several labels.
_CITATION_GROUP = re.compile(r"[\[(]\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*[\])]")


def build_prompt(question: str, hits: list[Hit]) -> str:
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
class Answer:
    question: str
    text: str = ""
    sources: list[Hit] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    found: bool = True

    @property
    def cited_hits(self) -> list[tuple[int, Hit]]:
        return [(n, self.sources[n - 1]) for n in self.cited]


class AnswerEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        k: int = 6,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        mode: Mode = Mode.HYBRID_RERANK,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_similarity = min_similarity
        self.mode = mode if retriever.reranker is not None else Mode.HYBRID

    def is_in_scope(self, question: str) -> bool:
        """Decline before calling the LLM when nothing in the corpus is semantically close."""
        return self.retriever.top_similarity(question) >= self.min_similarity

    def stream(self, question: str) -> tuple[Answer, Iterator[str]]:
        """Return the answer and its token stream; the answer is complete once the stream ends."""
        answer = Answer(question)
        if not question.strip() or not self.is_in_scope(question):
            answer.found = False
            answer.text = NOT_FOUND_MESSAGE
            return answer, iter([NOT_FOUND_MESSAGE])

        answer.sources = self.retriever.search(question, k=self.k, mode=self.mode)

        def tokens() -> Iterator[str]:
            parts: list[str] = []
            for token in self.llm.stream_text(
                SYSTEM_PROMPT, build_prompt(question, answer.sources)
            ):
                parts.append(token)
                yield token
            answer.text = "".join(parts)
            answer.cited, answer.invalid_citations = extract_citations(
                answer.text, len(answer.sources)
            )

        return answer, tokens()

    def ask(self, question: str) -> Answer:
        answer, tokens = self.stream(question)
        for _ in tokens:
            pass
        return answer
