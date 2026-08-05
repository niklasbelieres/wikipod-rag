from unittest.mock import MagicMock, patch

import numpy as np

from wikipod.rag.retriever import Retriever


def _fake_embedder(dimension: int = 4) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(dimension, dtype=np.float32)
    return embedder


def test_retrieve_converts_hits_into_chunks():
    hits = [
        {
            "score": 0.9,
            "article_id": 1,
            "article_title": "Climate change",
            "section_title": "Intro",
            "chunk_index": 0,
            "text": "Climate change refers to long-term shifts.",
        }
    ]
    with patch("wikipod.rag.retriever.knn_search", return_value=hits) as mock_search:
        retriever = Retriever(
            client=MagicMock(), index_name="wikipod-chunks", embedder=_fake_embedder(), top_k=5
        )
        chunks = retriever.retrieve("what is climate change?")

    assert len(chunks) == 1
    assert chunks[0].article_title == "Climate change"
    assert chunks[0].text == "Climate change refers to long-term shifts."
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert mock_search.call_args.args[1] == "wikipod-chunks"


def test_retrieve_uses_explicit_k_over_default_top_k():
    with patch("wikipod.rag.retriever.knn_search", return_value=[]) as mock_search:
        retriever = Retriever(
            client=MagicMock(), index_name="idx", embedder=_fake_embedder(), top_k=5
        )
        retriever.retrieve("query", k=2)

    assert mock_search.call_args.kwargs["k"] == 2 or mock_search.call_args.args[-1] == 2


def test_retrieve_falls_back_to_configured_top_k():
    with patch("wikipod.rag.retriever.knn_search", return_value=[]) as mock_search:
        retriever = Retriever(
            client=MagicMock(), index_name="idx", embedder=_fake_embedder(), top_k=7
        )
        retriever.retrieve("query")

    assert mock_search.call_args.kwargs.get("k") == 7
