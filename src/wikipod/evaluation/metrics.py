def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Anteil der relevant_ids, die unter den ersten k retrieved_ids auftauchen."""
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    found = len(set(top_k) & relevant_ids)
    return found / len(relevant_ids)


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
    
    ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant, strict=True) # Throws ValueError if len(all_retrived) != len(all_relevant)
    ]
        
    return sum(ranks) / len(ranks)