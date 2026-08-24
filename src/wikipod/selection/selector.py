"""Selects the subset of articles that fits a given storage budget.

The assignment asks for a subset that fits "a given storage budget" -- not a
fixed article count -- so selection is a budget-constrained problem, not a
simple top-N. This module treats it as a 0/1-knapsack instance and uses the
standard greedy heuristic: rank candidates by score-per-byte ("bang for the
buck") and take them in that order until the budget is exhausted. This isn't
provably optimal (true 0/1-knapsack is NP-hard) but is the standard, fast,
and well-justified approximation for a corpus this size.
"""

from collections.abc import Iterable

from wikipod.analysis.models import ArticleMetadata
from wikipod.config import SelectionWeights
from wikipod.selection.models import SelectionResult
from wikipod.selection.scoring import score_article

BYTES_PER_MB = 1024 * 1024


def select_within_budget(
    articles: Iterable[ArticleMetadata],
    link_frequencies: dict[str, int],
    weights: SelectionWeights,
    storage_budget_mb: float,
    pageviews: dict[str, int] | None = None,
) -> SelectionResult:
    """Greedily select articles by score-per-byte until `storage_budget_mb` is exhausted.

    `articles` only needs to be iterated once, so this accepts any iterable,
    not just a list -- notably a generator/streaming source (see
    `analysis.reader.stream_articles_metadata_cached`), which matters at
    full-corpus scale. Each candidate's `links`/`categories` are only needed
    to *compute* its score; the kept copy has them stripped immediately
    after, so this never holds `links` for the full corpus at once, only for
    whichever single article is currently being scored.
    """
    budget_bytes = storage_budget_mb * BYTES_PER_MB

    scored: list[tuple[float, ArticleMetadata]] = []
    total_candidates = 0
    for article in articles:
        total_candidates += 1
        score = score_article(article, link_frequencies, weights, pageviews) / max(
            article.html_size_bytes, 1
        )
        scored.append((score, article.model_copy(update={"links": [], "categories": []})))
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
        total_candidates=total_candidates,
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
