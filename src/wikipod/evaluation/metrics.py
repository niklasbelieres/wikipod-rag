import math


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Proportion of relevant_ids that appear among the first k retrieved_ids."""
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    found = len(set(top_k) & relevant_ids)
    return found / len(relevant_ids)

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Proportion of the first k retrieved results that are relevant."""
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(item in relevant_ids for item in top_k)
    return relevant_count / k

def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Evaluates how highly relevant results are ranked among the first k results."""
    if k <= 0 or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant_ids
    )

    ideal_relevant_count = min(len(relevant_ids), k)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg

def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """1/rank of the first relevant result in retrieved_ids, or 0.0 if none is found."""
    for rank, id in enumerate(retrieved_ids, start=1):
        if id in relevant_ids:
            return 1 / rank
        
    return 0.0
    


def mean_reciprocal_rank(all_retrieved: list[list], all_relevant: list[set]) -> float:
    """Average reciprocal rank across multiple queries (lists of equal length)."""
    if not all_retrieved:
        return 0.0
    
    # strict=True: raises ValueError if len(all_retrieved) != len(all_relevant)
    ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant, strict=True)
    ]
        
    return sum(ranks) / len(ranks)