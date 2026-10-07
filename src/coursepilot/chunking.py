"""Split pages into overlapping, sentence-aligned chunks that never cross a page boundary.

Keeping each chunk inside one page means every retrieved passage maps to exactly one
citation ("lecture3.pdf, p. 12"), which is what lets the evaluation score page accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from coursepilot.documents import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int  # 1-indexed page/slide/section in the file
    text: str
    label: str | None = None  # printed page number, if the document has one

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.label or self.page}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pack(sentences: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Greedily pack sentences into windows of at most `max_words` words.

    Consecutive windows share roughly `overlap_words` words of trailing context. A single
    sentence longer than `max_words` is split on word boundaries.
    """
    words_per_sentence: list[list[str]] = []
    for sentence in sentences:
        words = sentence.split()
        while len(words) > max_words:
            words_per_sentence.append(words[:max_words])
            words = words[max_words:]
        if words:
            words_per_sentence.append(words)

    windows: list[str] = []
    start = 0
    while start < len(words_per_sentence):
        end, count = start, 0
        while end < len(words_per_sentence) and (
            count + len(words_per_sentence[end]) <= max_words or end == start
        ):
            count += len(words_per_sentence[end])
            end += 1
        windows.append(" ".join(" ".join(s) for s in words_per_sentence[start:end]))
        if end >= len(words_per_sentence):
            break
        # Step back over trailing sentences to create overlap, but always make progress.
        back, overlap = end, 0
        while back - 1 > start and overlap + len(words_per_sentence[back - 1]) <= overlap_words:
            back -= 1
            overlap += len(words_per_sentence[back])
        start = back
    return windows


def chunk_pages(
    pages: list[Page], max_words: int = 180, overlap_words: int = 40, min_words: int = 5
) -> list[Chunk]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    chunks: list[Chunk] = []
    for page in pages:
        windows = _pack(split_sentences(page.text), max_words, overlap_words)
        for n, window in enumerate(w for w in windows if len(w.split()) >= min_words):
            chunks.append(
                Chunk(f"{page.source}#p{page.page}-{n}", page.source, page.page, window, page.label)
            )
    return chunks
