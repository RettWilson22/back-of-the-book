import io
from pathlib import Path

import pytest
from conftest import FakeEmbedder, make_pdf

from backofthebook import sample


def fake_download(content_from: Path, timeouts: list[float] | None = None):
    def urlopen(url: str, timeout: float) -> io.BytesIO:
        if timeouts is not None:
            timeouts.append(timeout)
        return io.BytesIO(content_from.read_bytes())

    return urlopen


def test_download_verifies_checksum_and_keeps_only_verified_files(tmp_path, monkeypatch):
    source = make_pdf(tmp_path / "book.pdf", ["Mean and median."])
    timeouts: list[float] = []
    monkeypatch.setattr(sample.urllib.request, "urlopen", fake_download(source, timeouts))
    monkeypatch.setattr(sample, "SHA256", sample.sha256(source))

    path = sample.download_sample(tmp_path / "corpus")

    assert path.name == sample.FILENAME
    assert list((tmp_path / "corpus").iterdir()) == [path]
    assert timeouts == [sample.DOWNLOAD_TIMEOUT]


def test_download_with_wrong_checksum_leaves_nothing_behind(tmp_path, monkeypatch):
    source = make_pdf(tmp_path / "book.pdf", ["Mean and median."])
    monkeypatch.setattr(sample.urllib.request, "urlopen", fake_download(source))

    with pytest.raises(sample.ChecksumError):
        sample.download_sample(tmp_path / "corpus")
    assert list((tmp_path / "corpus").iterdir()) == []


def test_interrupted_download_leaves_nothing_behind(tmp_path, monkeypatch):
    class Stalled(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise TimeoutError("the read operation timed out")

    monkeypatch.setattr(sample.urllib.request, "urlopen", lambda url, timeout: Stalled())

    with pytest.raises(TimeoutError):
        sample.download_sample(tmp_path / "corpus")
    assert list((tmp_path / "corpus").iterdir()) == []


def test_existing_verified_copy_is_not_downloaded_again(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    existing = make_pdf(corpus / sample.FILENAME, ["Cached."])
    monkeypatch.setattr(sample, "SHA256", sample.sha256(existing))
    monkeypatch.setattr(sample.urllib.request, "urlopen", pytest.fail)

    assert sample.download_sample(corpus) == existing


def test_build_sample_index_saves_a_loadable_index(tmp_path, monkeypatch):
    source = make_pdf(tmp_path / "book.pdf", ["The variance measures spread around the mean."])
    monkeypatch.setattr(sample.urllib.request, "urlopen", fake_download(source))
    monkeypatch.setattr(sample, "SHA256", sample.sha256(source))

    index = sample.build_sample_index(tmp_path / "corpus", tmp_path / "index", FakeEmbedder())

    assert index.sources == [sample.FILENAME]
    assert (tmp_path / "index" / "meta.json").exists()
