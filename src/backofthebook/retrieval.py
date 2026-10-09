"""Retrieval strategies: keyword (BM25), semantic (dense), hybrid (RRF), and reranked hybrid.

They share one interface so the evaluation can compare them on identical questions.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from backofthebook.chunking import Chunk
from backofthebook.index import CorpusIndex, Embedder

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Below this best-match cosine similarity, the materials don't cover the request. Chosen on
# eval/questions.jsonl and checked on eval/heldout.jsonl (see eval/README.md).
DEFAULT_MIN_SIMILARITY = 0.35

# Common English function words; removing them keeps BM25 focused on content terms.
STOPWORDS = frozenset(
    """
    a an and are as at be by can do does for from has have how i if in into is it its of on or
    that the their then there these this to was were what when where which who why will with
    you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Mode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None  # loaded on first use

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank_of_d), rank from 1.

    RRF only uses ranks, so it combines BM25 and cosine scores without calibrating them.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(
        self,
        index: CorpusIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ) -> None:
        if embedder.model_name != index.embedding_model:
            raise ValueError(
                f"index was built with {index.embedding_model}, not {embedder.model_name}"
            )
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        # How deep each ranking goes before fusion, and how many fused passages the reranker
        # scores. 20 matched 50 on both evaluation sets at about 40% of the rerank time.
        self.candidates = candidates
        self._bm25 = BM25Okapi([tokenize(c.text) or ["<empty>"] for c in index.chunks])
        self._last_query: tuple[str, np.ndarray] | None = None

    def _allowed(self, sources: Collection[str] | None) -> np.ndarray | None:
        """Boolean mask of chunks from `sources`, or None to allow every chunk."""
        if not sources:
            return None
        wanted = set(sources)
        return np.array([c.source in wanted for c in self.index.chunks])

    @staticmethod
    def _top(scores: np.ndarray, n: int, allowed: np.ndarray | None) -> list[tuple[int, float]]:
        order = np.argsort(-scores, kind="stable")
        if allowed is not None:
            order = order[allowed[order]]
        return [(int(i), float(scores[i])) for i in order[:n]]

    def _bm25_ranking(
        self, query: str, n: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        return [(i, s) for i, s in self._top(scores, n, allowed) if s > 0]

    def _query_vector(self, query: str) -> np.ndarray:
        """The query's embedding. The last one is remembered, since answering a question first
        checks whether the materials cover it and then searches, both with the same query.
        (Reading and replacing the tuple is safe when sessions share this retriever.)"""
        last = self._last_query
        if last is not None and last[0] == query:
            return last[1]
        vector: np.ndarray = self.embedder.encode([query])[0]
        self._last_query = (query, vector)
        return vector

    def _dense_ranking(
        self, query: str, n: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        return self._top(self.index.embeddings @ self._query_vector(query), n, allowed)

    def top_similarity(self, query: str, sources: Collection[str] | None = None) -> float:
        """Cosine similarity of the best-matching chunk; used to detect off-topic questions."""
        ranking = self._dense_ranking(query, 1, self._allowed(sources))
        return ranking[0][1] if ranking else 0.0

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = Mode.HYBRID_RERANK,
        sources: Collection[str] | None = None,
    ) -> list[Hit]:
        """Top-k chunks for `query`, optionally only from the given source files."""
        chunks = self.index.chunks
        allowed = self._allowed(sources)
        if mode is Mode.BM25:
            return [Hit(chunks[i], s) for i, s in self._bm25_ranking(query, k, allowed)]
        if mode is Mode.DENSE:
            return [Hit(chunks[i], s) for i, s in self._dense_ranking(query, k, allowed)]

        fused = reciprocal_rank_fusion(
            [
                [chunks[i].id for i, _ in self._bm25_ranking(query, self.candidates, allowed)],
                [chunks[i].id for i, _ in self._dense_ranking(query, self.candidates, allowed)],
            ]
        )
        if mode is Mode.HYBRID or self.reranker is None:
            if mode is Mode.HYBRID_RERANK:
                raise ValueError("hybrid+rerank mode needs a reranker")
            return [Hit(self.index.get(cid), s) for cid, s in fused[:k]]

        pool = [self.index.get(cid) for cid, _ in fused[: self.candidates]]
        scores = self.reranker.score(query, [c.text for c in pool])
        reranked = sorted(zip(pool, scores, strict=True), key=lambda pair: -pair[1])
        return [Hit(chunk, score) for chunk, score in reranked[:k]]
