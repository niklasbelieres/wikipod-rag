"""
Aggregate statistics over a collection of ArticleMetadata, used both for
exploratory analysis and for sanity-checking the selection step.
"""
from collections import Counter

from wikipod.analysis.models import ArticleMetadata


def summarize_articles(articles: list[ArticleMetadata]) -> dict[str, float | int]:
    """Return corpus-level summary stats (counts, averages, maxima)."""
    if not articles:
        return {}

    article_count = len(articles)
    word_counts = [a.word_count for a in articles]
    link_counts = [a.link_count for a in articles]
    sizes = [a.html_size_bytes for a in articles]

    return {
        "article_count": article_count,
        "avg_word_count": sum(word_counts) / article_count,
        "max_word_count": max(word_counts),
        "avg_link_count": sum(link_counts) / article_count,
        "max_link_count": max(link_counts),
        "total_size_mb": sum(sizes) / (1024 * 1024),
        "avg_size_kb": (sum(sizes) / article_count) / 1024,
    }


def link_frequency_map(articles: list[ArticleMetadata]) -> dict[str, int]:
    """Count how often each link target is referenced across the corpus.
    """
    counter: Counter[str] = Counter()
    for article in articles:
        counter.update(article.links)
    return dict(counter)


def most_common_links(articles: list[ArticleMetadata], n: int = 50) -> list[tuple[str, int]]:
    """Return the `n` most frequently linked-to targets in the corpus."""
    return Counter(link_frequency_map(articles)).most_common(n)
