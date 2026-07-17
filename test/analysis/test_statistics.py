from wikipod.analysis.models import ArticleMetadata
from wikipod.analysis.statistics import (
    link_frequency_map,
    most_common_links,
    summarize_articles,
)


def _make_article(article_id, word_count, link_count, links, size_bytes=1000):
    return ArticleMetadata(
        article_id=article_id,
        title=f"Article {article_id}",
        html_size_bytes=size_bytes,
        word_count=word_count,
        link_count=link_count,
        links=links,
        section_count=0,
        sections=[],
        categories=[],
    )


def test_summarize_articles_returns_empty_dict_for_no_articles():
    assert summarize_articles([]) == {}


def test_summarize_articles_computes_expected_aggregates():
    articles = [
        _make_article(1, word_count=100, link_count=2, links=["A", "B"], size_bytes=2048),
        _make_article(2, word_count=300, link_count=4, links=["A", "C"], size_bytes=4096),
    ]

    summary = summarize_articles(articles)

    assert summary["article_count"] == 2
    assert summary["avg_word_count"] == 200
    assert summary["max_word_count"] == 300
    assert summary["avg_link_count"] == 3
    assert summary["max_link_count"] == 4
    assert summary["total_size_mb"] == (2048 + 4096) / (1024 * 1024)


def test_link_frequency_map_counts_across_articles():
    articles = [
        _make_article(1, 10, 2, links=["A", "B"]),
        _make_article(2, 10, 2, links=["A", "C"]),
    ]

    freq = link_frequency_map(articles)

    assert freq == {"A": 2, "B": 1, "C": 1}


def test_most_common_links_orders_by_frequency():
    articles = [
        _make_article(1, 10, 3, links=["A", "A", "B"]),
        _make_article(2, 10, 1, links=["A"]),
    ]

    top = most_common_links(articles, n=2)

    assert top[0] == ("A", 3)
    assert top[1][0] == "B"
