"""Build, save, and load the search index: chunk metadata plus normalized dense embeddings.

Everything lives in a plain directory (`chunks.jsonl`, `embeddings.npy`, `meta.json`), so
there is no database server to run and an index can be inspected with standard tools.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from coursepilot.chunking import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

INDEX_FORMAT_VERSION = 1
USER_INDEX = Path(".coursepilot/index")  # where `coursepilot ingest` writes by default
BUNDLED_INDEX = Path("data/index")  # prebuilt index of the sample textbook, shipped in the repo
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of L2-normalized vectors."""
        ...


def default_index_dir(root: Path = Path(".")) -> Path:
    """Your own ingested index if there is one, otherwise the bundled sample index."""
    user = root / USER_INDEX
    return user if (user / "meta.json").exists() else root / BUNDLED_INDEX


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None  # loaded on first use

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 500
        )
        return np.asarray(vectors, dtype=np.float32)


class IndexBuildError(RuntimeError):
    pass


@dataclass
class CorpusIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray
    embedding_model: str

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.embeddings):
            raise IndexBuildError(
                f"{len(self.chunks)} chunks but {len(self.embeddings)} embeddings; "
                "rebuild the index"
            )
        self._by_id = {c.id: c for c in self.chunks}

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> CorpusIndex:
        if not chunks:
            raise IndexBuildError("no text could be extracted from the given documents")
        embeddings = embedder.encode([c.text for c in chunks])
        return cls(chunks, embeddings, embedder.model_name)

    def add(self, chunks: list[Chunk], embedder: Embedder) -> CorpusIndex:
        """Return a new index with `chunks` appended (replacing chunks from the same files)."""
        if embedder.model_name != self.embedding_model:
            raise IndexBuildError("cannot mix embedding models in one index")
        replaced = {c.source for c in chunks}
        keep = [i for i, c in enumerate(self.chunks) if c.source not in replaced]
        new = CorpusIndex.build(chunks, embedder)
        return CorpusIndex(
            [self.chunks[i] for i in keep] + new.chunks,
            np.vstack([self.embeddings[keep], new.embeddings]),
            self.embedding_model,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings)
        meta = {
            "format_version": INDEX_FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "num_chunks": len(self.chunks),
            "sources": self.sources,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> CorpusIndex:
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise IndexBuildError(f"no index at {directory}; run `coursepilot ingest` first")
        meta = json.loads(meta_path.read_text())
        if meta.get("format_version") != INDEX_FORMAT_VERSION:
            raise IndexBuildError("index was built by an incompatible version; rebuild it")
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        embeddings = np.load(directory / "embeddings.npy")
        return cls(chunks, embeddings, meta["embedding_model"])
