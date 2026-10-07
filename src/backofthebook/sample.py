"""The sample textbook used by the demo and the evaluation.

OpenStax "Principles of Data Science" (CC BY-NC-SA 4.0). The PDF is not committed to the
repository: it is downloaded from OpenStax and checked against the exact edition the
evaluation's page labels were written for.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from backofthebook.chunking import chunk_pages
from backofthebook.documents import load_document
from backofthebook.index import CorpusIndex, Embedder

URL = "https://assets.openstax.org/oscms-prodcms/media/documents/Principles-of-Data-Science-WEB.pdf"
SHA256 = "6b47205459a2f4e25a2b2a86369e674a26a5ae566a4cc428dd31047670692398"
FILENAME = "principles-of-data-science.pdf"


class ChecksumError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sample(directory: Path) -> Path:
    """Download the textbook into `directory` (skipped if a verified copy is already there)."""
    dest = directory / FILENAME
    if dest.exists() and sha256(dest) == SHA256:
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    urllib.request.urlretrieve(URL, partial)
    actual = sha256(partial)
    if actual != SHA256:
        partial.unlink()
        raise ChecksumError(
            f"downloaded file has checksum {actual}, not the edition the evaluation uses; "
            "OpenStax may have published an update"
        )
    partial.replace(dest)  # only a verified file ever appears under the final name
    return dest


def build_sample_index(corpus_dir: Path, index_dir: Path, embedder: Embedder) -> CorpusIndex:
    pdf = download_sample(corpus_dir)
    index = CorpusIndex.build(chunk_pages(load_document(pdf)), embedder)
    index.save(index_dir)
    return index
