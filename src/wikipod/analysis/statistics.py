"""
Aggregate statistics over a collection of ArticleMetadata, used both for
exploratory analysis and for sanity-checking the selection step.
"""
from collections import Counter
from collections.abc import Callable, Iterable

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


def link_frequency_map(
    articles: Iterable[ArticleMetadata],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Count how often each link target is referenced across the corpus.

    Accepts any iterable, not just a list -- notably a generator/streaming
    source (see `analysis.reader.stream_articles_metadata_cached`), so a
    full-corpus pass never needs every article's full metadata held in
    memory at once, just this function's own running counter.

    `on_progress`, if given, is called every 5000 articles as
    `on_progress(articles_processed, unique_link_targets_so_far)` --
    diagnostic hook to see whether the counter's *key count* keeps growing
    roughly linearly with corpus size (concerning: the dict itself could be
    the memory driver) or flattens out (the more expected shape, since
    popular link targets get "discovered" early and later articles mostly
    just bump existing counts rather than adding new keys).
    """
    counter: Counter[str] = Counter()
    for i, article in enumerate(articles, start=1):
        counter.update(article.links)
        if on_progress is not None and i % 5000 == 0:
            on_progress(i, len(counter))
    return dict(counter)


def most_common_links(articles: list[ArticleMetadata], n: int = 50) -> list[tuple[str, int]]:
    """Return the `n` most frequently linked-to targets in the corpus."""
    return Counter(link_frequency_map(articles)).most_common(n)
