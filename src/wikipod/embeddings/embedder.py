"""Embeds chunk text into dense vectors for indexing and retrieval.

Uses the sentence-transformers model configured in `config.embeddings.model_name`.
Indexing (`embed_chunks`) and retrieval (`embed_query`) go through the same
`Embedder` class so both sides of the vector search always live in the same
embedding space. Embeddings are L2-normalized so cosine similarity reduces to
a dot product, matching the `cosinesimil` space used by the OpenSearch index
(see `indexing/opensearch_client.py`).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from wikipod.chunking.models import Chunk


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Cache loaded models by name; loading is the expensive part."""
    return SentenceTransformer(model_name)


class Embedder:
    """Wraps a sentence-transformers model for chunk and query embedding."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = _load_model(model_name)

    @property
    def dimension(self) -> int:
        # `get_embedding_dimension` replaced `get_sentence_embedding_dimension` in
        # newer sentence-transformers; fall back for the >=3.0 range pyproject.toml allows.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        return get_dim() if get_dim else self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of raw strings. Returns an (n, dimension) float32 array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        return self.embed_texts([chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
