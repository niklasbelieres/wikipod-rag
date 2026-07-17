"""Selects the subset of articles that fits a given storage budget.

The assignment asks for a subset that fits "a given storage budget" -- not a
fixed article count -- so selection is a budget-constrained problem, not a
simple top-N. This module treats it as a 0/1-knapsack instance and uses the
standard greedy heuristic: rank candidates by score-per-byte ("bang for the
buck") and take them in that order until the budget is exhausted. This isn't
provably optimal (true 0/1-knapsack is NP-hard) but is the standard, fast,
and well-justified approximation for a corpus this size.
"""

from wikipod.analysis.models import ArticleMetadata
from wikipod.config import SelectionWeights
from wikipod.selection.models import SelectionResult
from wikipod.selection.scoring import score_article

BYTES_PER_MB = 1024 * 1024


def select_within_budget(
    articles: list[ArticleMetadata],
    link_frequencies: dict[str, int],
    weights: SelectionWeights,
    storage_budget_mb: float,
    pageviews: dict[str, int] | None = None,
) -> SelectionResult:
    """Greedily select articles by score-per-byte until `storage_budget_mb` is exhausted."""
    budget_bytes = storage_budget_mb * BYTES_PER_MB

    scored = [
        (
            score_article(a, link_frequencies, weights, pageviews) / max(a.html_size_bytes, 1),
            a,
        )
        for a in articles
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    selected: list[ArticleMetadata] = []
    used_bytes = 0

    for _, article in scored:
        if used_bytes + article.html_size_bytes > budget_bytes:
            continue
        selected.append(article)
        used_bytes += article.html_size_bytes

    return SelectionResult(
        selected=selected,
        budget_mb=storage_budget_mb,
        used_mb=used_bytes / BYTES_PER_MB,
        total_candidates=len(articles),
    )


def select_top_n(
    articles: list[ArticleMetadata],
    link_frequencies: dict[str, int],
    weights: SelectionWeights,
    n: int = 100,
    pageviews: dict[str, int] | None = None,
) -> list[ArticleMetadata]:
    """Rank-based selection without a storage constraint.

    Kept as a simpler alternative for quick experimentation/demos; the
    budget-aware `select_within_budget` above is what actually satisfies the
    assignment's "given storage budget" requirement.
    """
    return sorted(
        articles,
        key=lambda a: score_article(a, link_frequencies, weights, pageviews),
        reverse=True,
    )[:n]
