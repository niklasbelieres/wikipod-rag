import socket

import numpy as np
import pytest

from wikipod.chunking.models import Chunk
from wikipod.config import OpenSearchConfig
from wikipod.indexing.opensearch_client import (
    build_client,
    create_index,
    index_chunks,
    index_mapping,
    knn_search,
)


def _opensearch_available() -> bool:
    try:
        with socket.create_connection(("localhost", 9200), timeout=0.5):
            return True
    except OSError:
        return False


requires_opensearch = pytest.mark.skipif(
    not _opensearch_available(),
    reason="no OpenSearch reachable on localhost:9200; run `docker compose up -d`",
)


def _chunk(article_id: int, text: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        article_id=article_id,
        article_title=f"Article {article_id}",
        section_title="Intro",
        chunk_index=chunk_index,
        word_count=len(text.split()),
        text=text,
    )


def test_index_mapping_declares_knn_vector_with_requested_dimension():
    mapping = index_mapping(dimension=384)
    embedding_field = mapping["mappings"]["properties"]["embedding"]
    assert embedding_field["type"] == "knn_vector"
    assert embedding_field["dimension"] == 384
    assert mapping["settings"]["index"]["knn"] is True


def test_index_chunks_rejects_mismatched_lengths():
    chunks = [_chunk(1, "hello")]
    embeddings = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        index_chunks(client=None, index_name="x", chunks=chunks, embeddings=embeddings)


@requires_opensearch
def test_round_trip_index_and_search():
    config = OpenSearchConfig(index_name="wikipod-test-round-trip")
    client = build_client(config)
    create_index(client, config.index_name, dimension=4, recreate=True)

    chunks = [_chunk(1, "alpha"), _chunk(2, "beta")]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    indexed = index_chunks(client, config.index_name, chunks, embeddings)
    assert indexed == 2

    client.indices.refresh(index=config.index_name)
    hits = knn_search(client, config.index_name, embeddings[0], k=1)
    assert hits[0]["article_id"] == 1

    client.indices.delete(index=config.index_name)
