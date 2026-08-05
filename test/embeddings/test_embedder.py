import numpy as np
import pytest

from wikipod.chunking.models import Chunk
from wikipod.embeddings.embedder import Embedder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _model_available() -> bool:
    try:
        Embedder(MODEL_NAME)
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason=f"could not load '{MODEL_NAME}' (no network and not cached locally)",
)


@requires_model
def test_dimension_matches_model_output():
    embedder = Embedder(MODEL_NAME)
    vectors = embedder.embed_texts(["hello world"])
    assert vectors.shape == (1, embedder.dimension)


@requires_model
def test_embed_texts_returns_normalized_vectors():
    embedder = Embedder(MODEL_NAME)
    vectors = embedder.embed_texts(["Climate change", "Renewable energy"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@requires_model
def test_embed_texts_empty_list_returns_empty_array():
    embedder = Embedder(MODEL_NAME)
    vectors = embedder.embed_texts([])
    assert vectors.shape == (0, embedder.dimension)


@requires_model
def test_embed_query_returns_a_single_vector():
    embedder = Embedder(MODEL_NAME)
    vector = embedder.embed_query("What is climate change?")
    assert vector.shape == (embedder.dimension,)


@requires_model
def test_embed_chunks_matches_embed_texts():
    embedder = Embedder(MODEL_NAME)
    chunk = Chunk(
        article_id=1,
        article_title="Climate change",
        section_title="Intro",
        chunk_index=0,
        word_count=2,
        text="Climate change",
    )
    from_chunks = embedder.embed_chunks([chunk])
    from_texts = embedder.embed_texts(["Climate change"])
    np.testing.assert_allclose(from_chunks, from_texts)


@requires_model
def test_similar_texts_score_higher_than_unrelated_ones():
    embedder = Embedder(MODEL_NAME)
    query = embedder.embed_query("global warming and greenhouse gases")
    related = embedder.embed_query("climate change is driven by carbon emissions")
    unrelated = embedder.embed_query("recipe for chocolate chip cookies")

    assert np.dot(query, related) > np.dot(query, unrelated)
