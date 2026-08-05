"""Scores articles for the selection step.

Combines four signals, all log-dampened so no single very-large value (a
mega-popular link target, a huge article) dominates the score:

- word_count      -- how much substantive content the article has
- link_count       -- how well-connected the article is (outgoing)
- importance       -- sum of how often the article's outgoing links are
                       themselves referenced elsewhere in the corpus
- incoming_links    -- how often *this* article is linked to by others
                       (an in-corpus proxy for popularity)
- pageviews         -- real external pageview counts, if available (see
                       `selection/pageviews.py`); 0 otherwise

Weights are read from config (`config/default.yaml: selection.weights`)
rather than hardcoded, so they can be tuned and the choice documented in the
project report instead of living as invisible magic numbers.
"""

from __future__ import annotations

import math

from wikipod.analysis.models import ArticleMetadata
from wikipod.config import SelectionWeights
from wikipod.selection.pageviews import get_views


def score_article(
    article: ArticleMetadata,
    link_frequencies: dict[str, int],
    weights: SelectionWeights,
    pageviews: dict[str, int] | None = None,
) -> float:
    """Compute a single relevance/importance score for `article`."""
    pageviews = pageviews or {}

    word_score = math.log1p(article.word_count)
    link_score = math.log1p(article.link_count)
    importance_score = sum(math.log1p(link_frequencies.get(link, 0)) for link in article.links)
    incoming_links_score = math.log1p(link_frequencies.get(article.title, 0))
    pageviews_score = math.log1p(get_views(pageviews, article.title))

    return (
        weights.word_count * word_score
        + weights.link_count * link_score
        + weights.importance * importance_score
        + weights.incoming_links * incoming_links_score
        + weights.pageviews * pageviews_score
    )
