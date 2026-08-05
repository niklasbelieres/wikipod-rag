"""Retrieves the top-k most relevant chunks for a query via OpenSearch k-NN search.

Ties the embedding step (`embeddings.embedder.Embedder`) and the OpenSearch
client (`indexing.opensearch_client`) together so the rest of the RAG
pipeline only has to deal with a query string in, ranked `Chunk`s out.
"""

from __future__ import annotations

from opensearchpy import OpenSearch

from wikipod.chunking.models import Chunk
from wikipod.embeddings.embedder import Embedder
from wikipod.indexing.opensearch_client import knn_search


class Retriever:
    """Embeds a query and retrieves the top-k matching chunks from OpenSearch."""

    def __init__(
        self, client: OpenSearch, index_name: str, embedder: Embedder, top_k: int = 5
    ):
        self.client = client
        self.index_name = index_name
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        vector = self.embedder.embed_query(query)
        hits = knn_search(self.client, self.index_name, vector, k=k or self.top_k)
        return [_hit_to_chunk(hit) for hit in hits]


def _hit_to_chunk(hit: dict) -> Chunk:
    return Chunk(
        article_id=hit["article_id"],
        article_title=hit["article_title"],
        section_title=hit["section_title"],
        chunk_index=hit["chunk_index"],
        word_count=len(hit["text"].split()),
        text=hit["text"],
    )
