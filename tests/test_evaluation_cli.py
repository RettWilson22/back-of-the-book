import json
from pathlib import Path

import pytest
from conftest import FakeEmbedder, FakeReranker, make_pdf

from coursepilot import cli
from coursepilot.evaluation import (
    EvalQuestion,
    evaluate_mode,
    evaluate_scope,
    first_correct_rank,
    format_report,
    load_questions,
)
from coursepilot.retrieval import Mode, Retriever


def write_questions(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_questions_requires_gold_pages_for_answerable(tmp_path: Path):
    path = write_questions(
        tmp_path / "q.jsonl", [{"id": "q1", "question": "x", "answerable": True}]
    )
    with pytest.raises(ValueError, match="no gold pages"):
        load_questions(path)


def test_first_correct_rank(retriever: Retriever):
    hits = retriever.search("median sorted middle value", k=5, mode=Mode.HYBRID)
    assert first_correct_rank(hits, frozenset({("stats.pdf", 2)})) == 1
    assert first_correct_rank(hits, frozenset({("nope.pdf", 1)})) is None


def test_evaluate_mode_computes_recall_and_mrr(retriever: Retriever):
    questions = [
        EvalQuestion("a", "median middle value sorted", True, frozenset({("stats.pdf", 2)})),
        EvalQuestion("b", "k-means centroids", True, frozenset({("ml.pptx", 2)})),
        EvalQuestion("c", "something unrelated", True, frozenset({("missing.pdf", 9)})),
        EvalQuestion("d", "off topic", False),
    ]
    result = evaluate_mode(retriever, questions, Mode.HYBRID)

    assert result.recall[1] == pytest.approx(2 / 3)
    assert result.mrr == pytest.approx(2 / 3)
    assert result.misses == ["c"]


def test_evaluate_scope_counts_kept_and_declined(retriever: Retriever):
    questions = [
        EvalQuestion("a", "decision tree questions", True, frozenset({("ml.pptx", 1)})),
        EvalQuestion("b", "volcano eruption lava", False),
    ]
    low, high = evaluate_scope(retriever, questions, [0.0, 0.99])
    assert (low.answerable_kept, low.offtopic_declined) == (1.0, 0.0)
    assert high.offtopic_declined == 1.0


def test_format_report_has_one_row_per_mode(retriever: Retriever):
    questions = [EvalQuestion("a", "median", True, frozenset({("stats.pdf", 2)}))]
    results = [evaluate_mode(retriever, questions, m) for m in Mode]
    report = format_report(results, evaluate_scope(retriever, questions, [0.3]), 1, 0, "toy")
    assert report.count("\n| ") >= len(Mode) + 1
    assert "| hybrid+rerank |" in report


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CLI use the fake embedder/reranker instead of downloading real models."""

    class Embedder(FakeEmbedder):
        def __init__(self, model_name: str = "fake-embedder") -> None:
            self.model_name = "fake-embedder"

    monkeypatch.setattr("coursepilot.index.SentenceTransformerEmbedder", Embedder)
    monkeypatch.setattr("coursepilot.retrieval.CrossEncoderReranker", FakeReranker)
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "COURSEPILOT_LLM"):
        monkeypatch.delenv(var, raising=False)


def test_cli_ingest_ask_and_eval_end_to_end(tmp_path: Path, fake_models, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(
        docs / "stats.pdf",
        ["The variance measures spread around the mean value.", "Unrelated page."],
    )
    index = tmp_path / "index"

    assert cli.main(["--index", str(index), "ingest", str(docs)]) == 0
    assert (
        cli.main(["--index", str(index), "ask", "variance measures spread", "--llm", "extractive"])
        == 0
    )
    out = capsys.readouterr().out
    assert "variance measures spread" in out
    assert "[S1] stats.pdf, p. 1" in out

    questions = write_questions(
        tmp_path / "q.jsonl",
        [{"id": "q1", "question": "variance spread", "gold": [{"source": "stats.pdf", "page": 1}]}],
    )
    report = tmp_path / "report.md"
    assert (
        cli.main(
            ["--index", str(index), "eval", "--questions", str(questions), "--output", str(report)]
        )
        == 0
    )
    assert "| hybrid+rerank | 100.0% |" in report.read_text()


def test_cli_ingest_with_no_documents_fails(tmp_path: Path, fake_models):
    assert cli.main(["--index", str(tmp_path / "i"), "ingest", str(tmp_path)]) == 1


def test_cli_quiz_without_llm_reports_error(tmp_path: Path, fake_models, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Trees\nDecision trees split data on features.")
    index = tmp_path / "index"
    cli.main(["--index", str(index), "ingest", str(docs)])

    assert cli.main(["--index", str(index), "quiz", "decision trees"]) == 4
    assert "Error [NO_LLM_CONFIGURED]: Quiz generation needs an LLM" in capsys.readouterr().err
