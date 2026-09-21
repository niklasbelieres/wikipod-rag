from unittest.mock import MagicMock

import pytest

from wikipod.chunking.models import Chunk
from wikipod.evaluation.run_eval import retrieve_unique_chunks, run_eval


def chunk(title, index=0):
    return Chunk(
        article_id=1, article_title=title, section_title="Lead", chunk_index=index,
        word_count=1, text=str(index),
    )


def ranked_retriever(chunks):
    retriever = MagicMock()
    retriever.retrieve.side_effect = lambda query, k: chunks[:k]
    return retriever


def test_expands_multiple_times_and_keeps_first_chunk_per_article():
    candidates = [chunk("A", i) for i in range(30)] + [chunk("B"), chunk("C")]
    retriever = ranked_retriever(candidates)
    result = retrieve_unique_chunks(retriever, "query", 3)
    assert result == [candidates[0], candidates[30], candidates[31]]
    assert [call.kwargs["k"] for call in retriever.retrieve.call_args_list] == [12, 24, 48]


@pytest.mark.parametrize("size", [0, 5, 12])
def test_stops_when_backend_has_fewer_chunks_even_if_fewer_than_k_articles(size):
    retriever = ranked_retriever([chunk("A", i) for i in range(size)])
    result = retrieve_unique_chunks(retriever, "query", 3)
    assert [item.article_title for item in result] == (["A"] if size else [])
    assert retriever.retrieve.call_count == (2 if size == 12 else 1)


def test_candidate_limit_stops_repeated_full_responses_and_warns(caplog):
    retriever = ranked_retriever([chunk("A", i) for i in range(100)])
    result = retrieve_unique_chunks(retriever, "query", 3, max_candidates=25)
    assert len(result) == 1
    assert [call.kwargs["k"] for call in retriever.retrieve.call_args_list] == [12, 24, 25]
    assert "limit (25)" in caplog.text
    assert "found 1 of 3" in caplog.text


def test_initial_request_is_capped_and_success_does_not_warn(caplog):
    retriever = ranked_retriever([chunk("A"), chunk("B")])
    result = retrieve_unique_chunks(retriever, "query", 2, max_candidates=2)
    assert len(result) == 2
    retriever.retrieve.assert_called_once_with("query", k=2)
    assert not caplog.records


def test_uses_latest_ranking_instead_of_appending_new_articles():
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [chunk("A", i) for i in range(8)],
        [chunk("B"), chunk("A", 9)],
    ]
    result = retrieve_unique_chunks(retriever, "query", 2)
    assert [(item.article_title, item.chunk_index) for item in result] == [("B", 0), ("A", 9)]


def test_evaluation_scores_article_beyond_initial_candidate_window():
    retriever = ranked_retriever([chunk("A", i) for i in range(10)] + [chunk("Relevant")])
    result = run_eval(retriever, [{"query": "q", "relevant_titles": ["Relevant"]}], k=2)
    assert result["per_query"][0]["retrieved_titles"] == ["A", "Relevant"]
    assert result["mean_recall_at_k"] == 1.0
    assert result["mean_precision_at_k"] == 0.5
    assert result["mean_reciprocal_rank"] == 0.5


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -1}, {"k": 1, "max_candidates": 0}])
def test_invalid_limits_fail_before_retrieval(kwargs):
    retriever = MagicMock()
    with pytest.raises(ValueError):
        retrieve_unique_chunks(retriever, "query", **kwargs)
    retriever.retrieve.assert_not_called()
