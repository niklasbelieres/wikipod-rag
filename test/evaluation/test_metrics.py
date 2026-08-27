import pytest

from wikipod.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# -- recall_at_k --------------------------------------------------------------
def test_recall_at_k_finds_all_relevant_items():
    retrieved = [1, 4, 9, 8, 10]
    relevant = {1, 4, 9}
    k = 3

    assert recall_at_k(retrieved, relevant, k) == 1.0
    

def test_recall_at_k_finds_none():
    retrieved = [1,2,3,4,5]
    relevant = {6,7,8}
    k = 2
    
    assert recall_at_k(retrieved, relevant, k) == 0.0


def test_recall_at_k_finds_partial_match():
    retrieved = [1,2,3,4,5]
    relevant = {1,6,7,10}
    k = 3
    
    assert recall_at_k(retrieved, relevant, k) == 0.25



def test_recall_at_k_ignores_duplicate_matches():
    retrieved=[42, 42, 17]
    relevant={42}
    k=3
    
    assert recall_at_k(retrieved, relevant, k) == 1.0


def test_recall_at_k_only_considers_first_k_items():
    retrieved=[42, 43, 17]
    relevant={42,17}
    k=2
    
    assert recall_at_k(retrieved, relevant, k) == 0.5


def test_recall_at_k_handles_empty_relevant_ids():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_recall_at_k_handles_empty_retrieved_ids():
    assert recall_at_k([], {1, 2, 3}, k=3) == 0.0


def test_recall_at_k_handles_k_larger_than_retrieved_length():
    retrieved = [1, 2, 3]
    relevant = {1, 2, 3, 4, 5}
    # k=100 darf nicht ueber das Ende von retrieved hinaus IndexError werfen;
    # es zaehlen weiterhin nur die tatsaechlich vorhandenen 3 Treffer.
    assert recall_at_k(retrieved, relevant, k=100) == 0.6


def test_recall_at_k_handles_k_zero():
    assert recall_at_k([1, 2, 3], {1, 2}, k=0) == 0.0

# -- precision_at_k -----------------------------------------------------------
def test_precision_at_k_all_relevant():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_at_k_partial_match():
    assert precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == 0.5


def test_precision_at_k_no_match():
    assert precision_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_precision_at_k_handles_empty_retrieved_ids():
    assert precision_at_k([], {1, 2}, k=3) == 0.0


def test_precision_at_k_handles_k_zero():
    assert precision_at_k([1, 2, 3], {1, 2}, k=0) == 0.0


def test_precision_at_k_uses_requested_k_as_denominator():
    assert precision_at_k([1, 2], {1, 2}, k=5) == 0.4

# -- ndcg_at_k ----------------------------------------------------------------
def test_ndcg_at_k_perfect_ranking():
    retrieved = [1, 2, 3, 4]
    relevant = {1, 2}

    assert ndcg_at_k(retrieved, relevant, k=4) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_relevant_items_ranked_later():
    retrieved = [9, 8, 1, 2]
    relevant = {1, 2}

    assert 0.0 < ndcg_at_k(retrieved, relevant, k=4) < 1.0


def test_ndcg_at_k_no_relevant_results():
    assert ndcg_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_ndcg_at_k_handles_empty_relevant_ids():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_handles_empty_retrieved_ids():
    assert ndcg_at_k([], {1, 2}, k=3) == 0.0


def test_ndcg_at_k_handles_k_zero():
    assert ndcg_at_k([1, 2, 3], {1, 2}, k=0) == 0.0


def test_ndcg_at_k_only_considers_first_k_items():
    retrieved = [9, 1]
    relevant = {1}

    assert ndcg_at_k(retrieved, relevant, k=1) == 0.0
# -- reciprocal_rank ------------------------------------------------------------
def test_reciprocal_rank_first_position():
    assert reciprocal_rank([7, 1, 2, 3, 4], {7}) == 1.0


def test_reciprocal_rank_later_position():
    assert reciprocal_rank([17, 42, 99, 3, 8], {42}) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank([1, 2, 3, 4, 5], {99}) == 0.0


def test_reciprocal_rank_uses_earliest_match_when_multiple_relevant_present():
    # 17 und 99 sind beide relevant; 17 steht frueher (Position 2) als 99 (Position 3).
    assert reciprocal_rank([1, 17, 99, 2], {17, 99}) == 1 / 2


def test_reciprocal_rank_handles_empty_retrieved_ids():
    assert reciprocal_rank([], {1, 2, 3}) == 0.0


# -- mean_reciprocal_rank -------------------------------------------------------
def test_mean_reciprocal_rank_averages_across_queries():
    all_retrieved = [[17, 42, 99, 3, 8], [7, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
    all_relevant = [{42}, {7}, {99}]
    # reciprocal_rank pro Query: 0.5, 1.0, 0.0 -> Durchschnitt 0.5
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == 0.5


def test_mean_reciprocal_rank_single_query_matches_reciprocal_rank():
    # Der wertvollste Test hier -- hätte den Vertauschungs-Bug von eben sofort gefangen.
    retrieved = [17, 42, 99]
    relevant = {42}
    assert mean_reciprocal_rank([retrieved], [relevant]) == reciprocal_rank(retrieved, relevant)


def test_mean_reciprocal_rank_handles_empty_input():
    assert mean_reciprocal_rank([], []) == 0.0


def test_mean_reciprocal_rank_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([[1, 2], [3, 4]], [{1}])


def test_mean_reciprocal_rank_all_queries_perfect():
    all_retrieved = [[1, 2, 3], [4, 5, 6]]
    all_relevant = [{1}, {4}]
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == 1.0


def test_mean_reciprocal_rank_all_queries_miss():
    all_retrieved = [[1, 2, 3], [4, 5, 6]]
    all_relevant = [{99}, {100}]
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == 0.0
