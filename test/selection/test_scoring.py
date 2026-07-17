from wikipod.analysis.models import ArticleMetadata
from wikipod.config import SelectionWeights
from wikipod.selection.scoring import score_article


def _article(word_count=100, link_count=2, links=None, title="Article"):
    return ArticleMetadata(
        article_id=1,
        title=title,
        html_size_bytes=1000,
        word_count=word_count,
        link_count=link_count,
        links=links or [],
        section_count=0,
        sections=[],
        categories=[],
    )


def test_score_is_zero_for_empty_article_with_no_signal():
    article = _article(word_count=0, link_count=0, links=[])
    weights = SelectionWeights()
    assert score_article(article, link_frequencies={}, weights=weights) == 0.0


def test_more_words_and_links_increase_score():
    weights = SelectionWeights()
    small = _article(word_count=10, link_count=1)
    large = _article(word_count=1000, link_count=20)

    assert score_article(large, {}, weights) > score_article(small, {}, weights)


def test_incoming_links_increase_score():
    weights = SelectionWeights()
    article = _article(title="Popular")

    unlinked_score = score_article(article, link_frequencies={}, weights=weights)
    linked_score = score_article(article, link_frequencies={"Popular": 50}, weights=weights)

    assert linked_score > unlinked_score


def test_pageviews_match_despite_space_vs_underscore_title_format():
    # ZIM titles use spaces ("Climate change"); pageview dumps use
    # underscores ("Climate_change"). Scoring must match them anyway.
    article = _article(title="Climate change")
    pageviews = {"Climate_change": 5000}  # as load_pageviews() would key it
    weights = SelectionWeights(pageviews=1.0)

    score_with_dump_style_key = score_article(article, {}, weights, pageviews=pageviews)
    score_with_no_match = score_article(article, {}, weights, pageviews={})

    assert score_with_dump_style_key > score_with_no_match


def test_pageviews_only_affect_score_when_weight_is_nonzero():
    article = _article(title="Popular")
    pageviews = {"Popular": 10_000}

    zero_weight = SelectionWeights(pageviews=0.0)
    with_weight = SelectionWeights(pageviews=0.5)

    score_without = score_article(article, {}, zero_weight, pageviews=pageviews)
    score_with = score_article(article, {}, with_weight, pageviews=pageviews)

    assert score_with > score_without
