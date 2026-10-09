import pytest

from backofthebook.chunking import Chunk, chunk_pages, split_sentences
from backofthebook.documents import Page


def long_page(sentences: int, words_per_sentence: int = 10, page: int = 1) -> Page:
    text = " ".join(
        " ".join(f"w{s}_{w}" for w in range(words_per_sentence)) + "." for s in range(sentences)
    )
    return Page("doc.pdf", page, text)


def test_split_sentences_on_punctuation_and_newlines():
    assert split_sentences("One. Two! Three?\nFour") == ["One.", "Two!", "Three?", "Four"]


def test_chunks_respect_max_words():
    chunks = chunk_pages([long_page(40)], max_words=50, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 50 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_pages([long_page(20)], max_words=50, overlap_words=10)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[:10]


def test_every_sentence_is_covered():
    page = long_page(23)
    text = " ".join(c.text for c in chunk_pages([page], max_words=50, overlap_words=10))
    for sentence in split_sentences(page.text):
        assert sentence in text


def test_chunks_never_cross_pages_and_ids_are_unique():
    pages = [long_page(12, page=1), long_page(12, page=2)]
    chunks = chunk_pages(pages, max_words=40, overlap_words=10)
    assert {c.page for c in chunks} == {1, 2}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.id.startswith(f"doc.pdf#p{c.page}-") for c in chunks)


def test_sentence_longer_than_window_is_split():
    chunks = chunk_pages([long_page(1, words_per_sentence=120)], max_words=50, overlap_words=10)
    assert all(len(c.text.split()) <= 50 for c in chunks)
    assert sum(len(c.text.split()) for c in chunks) >= 120


def test_tiny_fragments_are_dropped():
    chunks = chunk_pages(
        [Page("doc.pdf", 1, "Figure 2."), Page("doc.pdf", 2, "This is a real sentence here.")]
    )
    assert [c.page for c in chunks] == [2]


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_pages([long_page(5)], max_words=20, overlap_words=20)


def test_citation_prefers_printed_page_label():
    assert Chunk("id", "book.pdf", 21, "text", label="11").citation == "book.pdf, p. 11"
    assert Chunk("id", "deck.pptx", 4, "text").citation == "deck.pptx, p. 4"


def test_chunks_stay_short_even_without_spaces():
    """A word is only a run of non-spaces, so one huge "word" could make a huge passage."""
    chunks = chunk_pages([Page("blob.txt", 1, "start of text " + "x" * 50_000 + " end")])

    assert chunks
    assert max(len(c.text) for c in chunks) <= 2_000
    assert sum(len(c.text.replace(" ", "")) for c in chunks) >= 50_000


def test_window_character_limit_keeps_ordinary_text_unchanged():
    page = long_page(23)
    assert chunk_pages([page]) == chunk_pages([page], max_chars=100_000)
