from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder, FakeReranker

from backofthebook.chunking import Chunk, chunk_pages
from backofthebook.documents import Page
from backofthebook.index import CorpusIndex, IndexBuildError
from backofthebook.retrieval import Mode, Retriever, reciprocal_rank_fusion, tokenize


def test_index_round_trips_through_disk(index: CorpusIndex, tmp_path: Path):
    index.save(tmp_path / "idx")
    loaded = CorpusIndex.load(tmp_path / "idx")

    assert loaded.chunks == index.chunks
    assert np.array_equal(loaded.embeddings, index.embeddings)
    assert loaded.embedding_model == "fake-embedder"


def test_loading_missing_index_explains_what_to_do(tmp_path: Path):
    with pytest.raises(IndexBuildError, match="backofthebook ingest"):
        CorpusIndex.load(tmp_path)


def test_index_rejects_mismatched_embeddings(chunks: list[Chunk]):
    with pytest.raises(IndexBuildError):
        CorpusIndex(chunks, np.zeros((1, 4), dtype=np.float32), "m")


def test_building_from_no_chunks_fails_clearly():
    with pytest.raises(IndexBuildError, match="no text"):
        CorpusIndex.build([], FakeEmbedder())


def test_merge_replaces_chunks_from_the_same_file(index: CorpusIndex):
    newer = CorpusIndex.build(
        [Chunk("ml.pptx#p1-0", "ml.pptx", 1, "Neural networks.")], FakeEmbedder()
    )
    updated = CorpusIndex.merge([index, newer])

    ml = [c for c in updated.chunks if c.source == "ml.pptx"]
    assert [c.text for c in ml] == ["Neural networks."]
    assert len(updated.chunks) == len(updated.embeddings)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Mean of X?") == ["mean", "x"]


def test_rrf_matches_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0][0] == "b"


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_finds_the_right_page(retriever: Retriever, mode: Mode):
    hits = retriever.search("how is the median computed from sorted data", k=3, mode=mode)
    assert (hits[0].chunk.source, hits[0].chunk.page) == ("stats.pdf", 2)


def test_rerank_mode_uses_the_reranker(index: CorpusIndex):
    reranker = FakeReranker()
    retriever = Retriever(index, FakeEmbedder(), reranker)
    retriever.search("k-means centroids", mode=Mode.HYBRID_RERANK)
    assert reranker.calls == 1


def test_rerank_mode_without_reranker_is_an_error(index: CorpusIndex):
    with pytest.raises(ValueError, match="reranker"):
        Retriever(index, FakeEmbedder()).search("mean", mode=Mode.HYBRID_RERANK)


def test_bm25_returns_nothing_for_unknown_words(retriever: Retriever):
    assert retriever.search("zzzz qqqq", mode=Mode.BM25) == []


def test_retriever_refuses_a_different_embedding_model(index: CorpusIndex):
    class OtherEmbedder(FakeEmbedder):
        model_name = "other"

    with pytest.raises(ValueError, match="built with"):
        Retriever(index, OtherEmbedder())


def test_top_similarity_is_higher_for_on_topic_questions(retriever: Retriever):
    assert retriever.top_similarity("decision tree questions") > retriever.top_similarity(
        "volcano eruption lava"
    )


def test_default_index_prefers_the_users_own_index(tmp_path: Path, index: CorpusIndex):
    from backofthebook.index import BUNDLED_INDEX, USER_INDEX, default_index_dir

    assert default_index_dir(tmp_path) == tmp_path / BUNDLED_INDEX
    index.save(tmp_path / USER_INDEX)
    assert default_index_dir(tmp_path) == tmp_path / USER_INDEX


def test_bundled_sample_index_is_valid():
    root = Path(__file__).resolve().parents[1]
    bundled = CorpusIndex.load(root / "data" / "index")

    assert bundled.sources == ["principles-of-data-science.pdf"]
    assert len(bundled.chunks) == 1479
    assert bundled.embeddings.shape == (1479, 384)
    assert np.allclose((bundled.embeddings**2).sum(axis=1), 1.0, atol=1e-4)
    assert all(c.label for c in bundled.chunks if c.page > 10)  # printed page numbers detected


def test_adding_uploads_never_changes_the_shared_index(index: CorpusIndex):
    """The sample index is shared by every visitor; one visitor's uploads must only ever
    produce a new, private index for that visitor's session."""
    before_chunks, before_vectors = list(index.chunks), index.embeddings.copy()

    upload = CorpusIndex.build(
        [Chunk("secret.pdf#p1-0", "secret.pdf", 1, "Private notes.")], FakeEmbedder()
    )
    private = CorpusIndex.merge([index, upload])

    assert "secret.pdf" in private.sources
    assert "secret.pdf" not in index.sources
    assert index.chunks == before_chunks
    assert np.array_equal(index.embeddings, before_vectors)


def test_merge_combines_without_reembedding_and_later_parts_win(index: CorpusIndex):
    mine = CorpusIndex.build(
        [Chunk("notes.md#p1-0", "notes.md", 1, "My own notes.")], FakeEmbedder()
    )
    newer = CorpusIndex.build(
        [Chunk("notes.md#p1-0", "notes.md", 1, "Edited notes.")], FakeEmbedder()
    )

    merged = CorpusIndex.merge([index, mine, newer])

    assert set(merged.sources) == set(index.sources) | {"notes.md"}
    assert [c.text for c in merged.chunks if c.source == "notes.md"] == ["Edited notes."]
    assert len(merged.chunks) == len(merged.embeddings) == len(index.chunks) + 1
    assert "notes.md" not in index.sources  # parts are untouched


def test_reranker_scores_at_most_twenty_candidates():
    pages = [
        Page("big.pdf", n, f"Page {n} talks about the mean and median of data.") for n in range(60)
    ]
    index = CorpusIndex.build(chunk_pages(pages), FakeEmbedder())
    scored: list[int] = []

    class CountingReranker(FakeReranker):
        def score(self, query: str, texts: list[str]) -> list[float]:
            scored.append(len(texts))
            return super().score(query, texts)

    Retriever(index, FakeEmbedder(), CountingReranker()).search("mean median", k=5)
    assert scored == [20]


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return super().encode(texts)


def test_a_question_is_embedded_once_for_the_scope_check_and_the_search(index: CorpusIndex):
    from conftest import FakeLLM

    from backofthebook.answer import AnswerEngine

    embedder = CountingEmbedder()
    retriever = Retriever(index, embedder, FakeReranker())
    engine = AnswerEngine(retriever, FakeLLM("A [S1]."), min_similarity=0.0)

    engine.ask("what is the median of sorted data")
    engine.ask("what is k-means clustering")

    assert embedder.calls == [["what is the median of sorted data"], ["what is k-means clustering"]]
