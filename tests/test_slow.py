"""Tests with the real embedding and reranking models (downloaded on first run).

Run with `pytest -m slow`. The textbook test also needs `python scripts/download_corpus.py`.
"""

from pathlib import Path

import pytest
from conftest import TOY_PAGES

from backofthebook.chunking import chunk_pages
from backofthebook.evaluation import evaluate_mode, load_questions
from backofthebook.index import CorpusIndex, SentenceTransformerEmbedder
from backofthebook.retrieval import CrossEncoderReranker, Mode, Retriever

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
TEXTBOOK = ROOT / "data" / "corpus" / "principles-of-data-science.pdf"


@pytest.fixture(scope="module")
def real_retriever() -> Retriever:
    embedder = SentenceTransformerEmbedder()
    index = CorpusIndex.build(chunk_pages(TOY_PAGES), embedder)
    return Retriever(index, embedder, CrossEncoderReranker())


def test_real_embeddings_are_normalized(real_retriever: Retriever):
    norms = (real_retriever.index.embeddings**2).sum(axis=1)
    assert norms == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    ("question", "page"),
    [
        ("Which statistic is the center value once you order the numbers?", ("stats.pdf", 2)),
        ("What does it mean when a model learns the noise in its training set?", ("ml.pptx", 3)),
    ],
)
def test_paraphrased_questions_find_the_right_page(real_retriever: Retriever, question, page):
    hit = real_retriever.search(question, k=1, mode=Mode.HYBRID_RERANK)[0]
    assert (hit.chunk.source, hit.chunk.page) == page


def test_off_topic_scores_below_on_topic(real_retriever: Retriever):
    assert real_retriever.top_similarity("Who painted the Mona Lisa?") < 0.35
    assert real_retriever.top_similarity("How does k-means pick cluster centers?") > 0.35


@pytest.mark.skipif(not TEXTBOOK.exists(), reason="run scripts/download_corpus.py first")
def test_textbook_retrieval_does_not_regress(tmp_path: Path):
    """Guards the published numbers: fails if hybrid+rerank Recall@10 drops below 95%."""
    from backofthebook.documents import load_document

    embedder = SentenceTransformerEmbedder()
    index = CorpusIndex.build(chunk_pages(load_document(TEXTBOOK)), embedder)
    retriever = Retriever(index, embedder, CrossEncoderReranker())
    result = evaluate_mode(
        retriever, load_questions(ROOT / "eval" / "questions.jsonl"), Mode.HYBRID_RERANK
    )
    assert result.recall[10] >= 0.95
    assert result.mrr >= 0.80
