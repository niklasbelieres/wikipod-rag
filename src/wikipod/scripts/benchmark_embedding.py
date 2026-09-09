"""Benchmarks the configured embedding model in isolation from OpenSearch.

Measures two distinct numbers the report asks for (see chapter 4, indexing):
  - bulk throughput (chunks/second), matching the batch encode() used during
    `wikipod index` for the whole corpus
  - single-query latency (ms/query), matching the batch-size-1 encode() used
    on the retrieval hot path in `rag/retriever.py`

Deliberately does not touch the ZIM, OpenSearch, or the real chunking
pipeline -- this isolates model inference time from I/O/network overhead, so
the numbers reflect the embedding model's own cost on this hardware, not the
rest of the indexing pipeline (see `measure_corpus_size.py`/`analyze_scoring.py`
for corpus-dependent numbers instead).

Usage:
    WIKIPOD_ENV=server python -m wikipod.scripts.benchmark_embedding
"""

from __future__ import annotations

import random
import time

from wikipod.config import get_config
from wikipod.embeddings.embedder import Embedder

N_BULK_CHUNKS = 1000
N_QUERY_REPS = 30

# Rough English word list to build synthetic chunk text from -- exact content
# doesn't matter for a throughput benchmark, only realistic length does.
_WORDS = (
    "wikipedia article history politics science culture technology society "
    "government economy language people place event organization concept"
).split()


def _fake_text(word_count: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(word_count))


def main() -> None:
    config = get_config()
    embedder = Embedder(config.embeddings.model_name, batch_size=config.embeddings.batch_size)
    rng = random.Random(0)

    print(f"Model: {config.embeddings.model_name}  (batch_size={embedder.batch_size})")

    # -- Bulk throughput (chunks/second), matching config.chunking.max_words --
    chunk_texts = [
        _fake_text(config.chunking.max_words, rng) for _ in range(N_BULK_CHUNKS)
    ]

    embedder.embed_texts(chunk_texts[: embedder.batch_size])  # warmup, discarded

    start = time.perf_counter()
    embedder.embed_texts(chunk_texts)
    elapsed = time.perf_counter() - start

    print(
        f"\nBulk embedding: {N_BULK_CHUNKS} chunks (~{config.chunking.max_words} words each) "
        f"in {elapsed:.2f}s -> {N_BULK_CHUNKS / elapsed:.1f} chunks/s"
    )

    # -- Single-query latency (ms/query), batch size 1 like retrieval time --
    sample_query = _fake_text(8, rng)

    embedder.embed_query(sample_query)  # warmup

    start = time.perf_counter()
    for _ in range(N_QUERY_REPS):
        embedder.embed_query(sample_query)
    elapsed = time.perf_counter() - start

    print(
        f"Single-query embedding: {N_QUERY_REPS} reps -> "
        f"{elapsed / N_QUERY_REPS * 1000:.1f} ms/query"
    )


if __name__ == "__main__":
    main()
