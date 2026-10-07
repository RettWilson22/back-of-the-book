"""Measure retrieval quality and off-topic detection against a labeled question set.

Each question lists the page(s) that answer it. A retrieved chunk counts as correct when it
comes from one of those pages, so the metrics answer the question a student cares about:
"did the assistant look at the right page?"

- Recall@k: share of answerable questions with a correct page in the top k results.
- MRR@10: mean of 1/rank of the first correct page (0 if not in the top 10).
- Scope accuracy: answerable questions kept in scope, off-topic questions declined.

No LLM is involved, so results are deterministic and free to reproduce.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from coursepilot.retrieval import Hit, Mode, Retriever

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    gold_pages: frozenset[tuple[str, int]] = field(default_factory=frozenset)
    topic: str = ""


def load_questions(path: Path) -> list[EvalQuestion]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            gold = frozenset((g["source"], int(g["page"])) for g in row.get("gold", []))
            answerable = bool(row.get("answerable", True))
            if answerable and not gold:
                raise ValueError(f"{path}:{line_number}: answerable question has no gold pages")
            questions.append(
                EvalQuestion(row["id"], row["question"], answerable, gold, row.get("topic", ""))
            )
    return questions


def first_correct_rank(hits: list[Hit], gold: frozenset[tuple[str, int]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if (hit.chunk.source, hit.chunk.page) in gold:
            return rank
    return None


@dataclass
class ModeResult:
    mode: Mode
    recall: dict[int, float]
    mrr: float
    median_latency_ms: float
    misses: list[str]  # ids of questions with no correct page in the top 10


def evaluate_mode(retriever: Retriever, questions: list[EvalQuestion], mode: Mode) -> ModeResult:
    answerable = [q for q in questions if q.answerable]
    if not answerable:
        raise ValueError("no answerable questions to evaluate")
    hits_at = dict.fromkeys(KS, 0)
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    misses: list[str] = []
    for q in answerable:
        start = time.perf_counter()
        hits = retriever.search(q.question, k=max(KS), mode=mode)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_correct_rank(hits, q.gold_pages)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank is None:
            misses.append(q.id)
        for k in KS:
            hits_at[k] += rank is not None and rank <= k
    n = len(answerable)
    return ModeResult(
        mode,
        {k: hits_at[k] / n for k in KS},
        sum(reciprocal_ranks) / n,
        statistics.median(latencies),
        misses,
    )


@dataclass
class ScopeResult:
    threshold: float
    answerable_kept: float  # share of answerable questions not wrongly declined
    offtopic_declined: float  # share of off-topic questions correctly declined


def evaluate_scope(
    retriever: Retriever, questions: list[EvalQuestion], thresholds: list[float]
) -> list[ScopeResult]:
    answerable = [retriever.top_similarity(q.question) for q in questions if q.answerable]
    offtopic = [retriever.top_similarity(q.question) for q in questions if not q.answerable]
    results = []
    for t in thresholds:
        kept = sum(s >= t for s in answerable) / len(answerable) if answerable else 0.0
        declined = sum(s < t for s in offtopic) / len(offtopic) if offtopic else 0.0
        results.append(ScopeResult(t, kept, declined))
    return results


def format_report(
    mode_results: list[ModeResult],
    scope_results: list[ScopeResult],
    num_answerable: int,
    num_offtopic: int,
    corpus: str,
) -> str:
    lines = [
        f"Corpus: {corpus}. Questions: {num_answerable} answerable, {num_offtopic} off-topic.",
        "",
        "| Retrieval | " + " | ".join(f"Recall@{k}" for k in KS) + " | MRR@10 | Median latency |",
        "|---|" + "---|" * (len(KS) + 2),
    ]
    for r in mode_results:
        recalls = " | ".join(f"{r.recall[k]:.1%}" for k in KS)
        lines.append(f"| {r.mode.value} | {recalls} | {r.mrr:.3f} | {r.median_latency_ms:.0f} ms |")
    lines += [
        "",
        "| Off-topic threshold | Answerable kept | Off-topic declined |",
        "|---|---|---|",
    ]
    for s in scope_results:
        lines.append(f"| {s.threshold:.2f} | {s.answerable_kept:.1%} | {s.offtopic_declined:.1%} |")
    return "\n".join(lines) + "\n"
