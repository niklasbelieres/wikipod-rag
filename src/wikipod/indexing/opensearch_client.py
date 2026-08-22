"""OpenSearch client for indexing and searching embedded chunks.

Wraps the low-level `opensearch-py` client behind the three operations the
pipeline needs: create the k-NN index once (`create_index`), bulk-index
`Chunk`s together with their pre-computed vectors (`index_chunks`), and run a
k-NN search for retrieval (`knn_search`). Kept separate from `rag/retriever.py`
so the OpenSearch-specific wire format doesn't leak into the retrieval API.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from opensearchpy import OpenSearch, helpers

from wikipod.chunking.models import Chunk
from wikipod.config import OpenSearchConfig

logger = logging.getLogger(__name__)


def build_client(config: OpenSearchConfig) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": config.host, "port": config.port}],
        http_compress=True,
        use_ssl=config.use_ssl,
        verify_certs=config.verify_certs,
        timeout=config.timeout,
    )


def index_mapping(dimension: int) -> dict[str, Any]:
    """k-NN index mapping: HNSW over cosine similarity for normalized vectors."""
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "article_id": {"type": "integer"},
                "article_title": {"type": "keyword"},
                "section_title": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
            }
        },
    }


def create_index(
    client: OpenSearch, index_name: str, dimension: int, recreate: bool = False
) -> None:
    """Create the k-NN index if it doesn't exist yet (or drop and recreate it)."""
    exists = client.indices.exists(index=index_name)
    if exists and recreate:
        client.indices.delete(index=index_name)
        exists = False
    if not exists:
        client.indices.create(index=index_name, body=index_mapping(dimension))
        logger.info("Created OpenSearch index '%s' (dim=%d)", index_name, dimension)


def index_chunks(
    client: OpenSearch, index_name: str, chunks: list[Chunk], embeddings: np.ndarray
) -> int:
    """Bulk-index `chunks` with their pre-computed `embeddings`. Returns the count indexed."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    actions = (
        {
            "_index": index_name,
            "_id": chunk.chunk_id,
            "_source": {
                "text": chunk.text,
                "article_id": chunk.article_id,
                "article_title": chunk.article_title,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "embedding": vector.tolist(),
            },
        }
        for chunk, vector in zip(chunks, embeddings, strict=True)
    )
    success, errors = helpers.bulk(client, actions, stats_only=False)
    if errors:
        logger.warning("%d chunks failed to index", len(errors))
    return success


def knn_search(
    client: OpenSearch, index_name: str, query_vector: np.ndarray, k: int = 5
) -> list[dict[str, Any]]:
    """Run a k-NN search and return hit sources with their similarity score."""
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": query_vector.tolist(), "k": k}}},
    }
    response = client.search(index=index_name, body=body)
    return [{"score": hit["_score"], **hit["_source"]} for hit in response["hits"]["hits"]]
