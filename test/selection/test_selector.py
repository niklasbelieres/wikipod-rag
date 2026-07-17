from wikipod.analysis.models import ArticleMetadata
from wikipod.config import SelectionWeights
from wikipod.selection.selector import select_top_n, select_within_budget


def _article(article_id, size_bytes, word_count=100, title=None):
    return ArticleMetadata(
        article_id=article_id,
        title=title or f"Article {article_id}",
        html_size_bytes=size_bytes,
        word_count=word_count,
        link_count=0,
        links=[],
        section_count=0,
        sections=[],
        categories=[],
    )


def test_select_within_budget_never_exceeds_the_budget():
    one_mb = 1024 * 1024
    articles = [_article(i, size_bytes=int(0.4 * one_mb)) for i in range(5)]

    result = select_within_budget(
        articles, link_frequencies={}, weights=SelectionWeights(), storage_budget_mb=1.0
    )

    assert result.used_mb <= result.budget_mb
    # 0.4MB articles: at most 2 fit in a 1MB budget
    assert len(result.selected) <= 2


def test_select_within_budget_prefers_higher_scoring_articles_of_equal_size():
    one_mb = 1024 * 1024
    low_value = _article(1, size_bytes=one_mb, word_count=1, title="Low")
    high_value = _article(2, size_bytes=one_mb, word_count=10_000, title="High")

    result = select_within_budget(
        [low_value, high_value],
        link_frequencies={},
        weights=SelectionWeights(),
        storage_budget_mb=1.0,  # only room for one of them
    )

    assert len(result.selected) == 1
    assert result.selected[0].title == "High"


def test_select_within_budget_reports_candidate_count():
    articles = [_article(i, size_bytes=100) for i in range(10)]
    result = select_within_budget(
        articles, link_frequencies={}, weights=SelectionWeights(), storage_budget_mb=100
    )
    assert result.total_candidates == 10


def test_select_top_n_respects_the_limit():
    articles = [_article(i, size_bytes=100) for i in range(10)]
    selected = select_top_n(articles, link_frequencies={}, weights=SelectionWeights(), n=3)
    assert len(selected) == 3
