import math


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Anteil der relevant_ids, die unter den ersten k retrieved_ids auftauchen."""
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    found = len(set(top_k) & relevant_ids)
    return found / len(relevant_ids)

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Anteil der ersten k Treffer, die relevant sind."""
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(item in relevant_ids for item in top_k)
    return relevant_count / k

def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Bewertet, wie weit oben relevante Treffer unter den ersten k Ergebnissen stehen."""
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
    """1/Rang des ersten relevanten Treffers in retrieved_ids, 0.0 falls keiner drin ist."""
    for rank, id in enumerate(retrieved_ids, start=1):
        if id in relevant_ids:
            return 1 / rank
        
    return 0.0
    


def mean_reciprocal_rank(all_retrieved: list[list], all_relevant: list[set]) -> float:
    """Durchschnitt von reciprocal_rank über mehrere Queries (Listen gleicher Länge)."""
    if not all_retrieved:
        return 0.0
    
    # strict=True: raises ValueError if len(all_retrieved) != len(all_relevant)
    ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant, strict=True)
    ]
        
    return sum(ranks) / len(ranks)