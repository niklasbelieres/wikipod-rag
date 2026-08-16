from pathlib import Path
from unittest.mock import patch

import pytest

from wikipod.analysis.metadata import extract_metadata
from wikipod.analysis.models import ArticleMetadata
from wikipod.analysis.reader import (
    is_html_redirect,
    iter_articles,
    read_articles_metadata_cached,
    read_articles_metadata_parallel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIM_FILE = PROJECT_ROOT / "test" / "data" / "climate-change-mini.zim"

requires_zim = pytest.mark.skipif(
    not ZIM_FILE.exists(), reason="demo ZIM file not present; see test/data/README.md"
)


def test_detects_nonexistent_file_path():
    with pytest.raises(FileNotFoundError):
        next(iter_articles("does_not_exist.zim"))


def test_detects_wrong_file_type():
    with pytest.raises(ValueError):
        next(iter_articles(__file__))


def test_detects_html_redirect():
    html = """
    <html><head>
      <meta http-equiv="refresh" content="0;URL='./Target'" />
    </head></html>
    """
    assert is_html_redirect(html) is True


def test_detects_normal_article():
    html = "<html><body><h1>Climate change</h1></body></html>"
    assert is_html_redirect(html) is False


@requires_zim
def test_reader_returns_articles():
    articles = list(_take(iter_articles(ZIM_FILE), 5))
    assert len(articles) > 0


@requires_zim
def test_articles_have_titles_and_html():
    article = next(iter_articles(ZIM_FILE))
    assert isinstance(article.title, str) and article.title
    assert isinstance(article.html, str) and article.html
    assert "<html" in article.html.lower()


@requires_zim
def test_reader_skips_html_redirects():
    for article in _take(iter_articles(ZIM_FILE), 50):
        assert not is_html_redirect(article.html)


@requires_zim
def test_reader_returns_unique_article_ids():
    ids = [a.article_id for a in _take(iter_articles(ZIM_FILE), 100)]
    assert len(ids) == len(set(ids))


def _take(iterator, n):
    for i, item in enumerate(iterator):
        if i >= n:
            break
        yield item


def _fake_article_metadata() -> list[ArticleMetadata]:
    return [
        ArticleMetadata(
            article_id=1,
            title="A",
            html_size_bytes=10,
            word_count=1,
            link_count=0,
            links=[],
            section_count=0,
            sections=[],
            categories=[],
        )
    ]


def test_read_articles_metadata_cached_parses_and_writes_cache_on_first_call(tmp_path):
    zim_path = tmp_path / "fake.zim"
    zim_path.write_bytes(b"fake")
    cache_path = tmp_path / "cache.pkl"
    fake_articles = _fake_article_metadata()

    with patch(
        "wikipod.analysis.reader.read_articles_metadata_parallel", return_value=fake_articles
    ) as mock_parse:
        result = read_articles_metadata_cached(zim_path, cache_path)

    assert result == fake_articles
    assert cache_path.exists()
    mock_parse.assert_called_once()


def test_read_articles_metadata_cached_reuses_cache_on_second_call(tmp_path):
    zim_path = tmp_path / "fake.zim"
    zim_path.write_bytes(b"fake")
    cache_path = tmp_path / "cache.pkl"
    fake_articles = _fake_article_metadata()

    with patch(
        "wikipod.analysis.reader.read_articles_metadata_parallel", return_value=fake_articles
    ) as mock_parse:
        read_articles_metadata_cached(zim_path, cache_path)
        result = read_articles_metadata_cached(zim_path, cache_path)

    assert result == fake_articles
    mock_parse.assert_called_once()


def test_read_articles_metadata_cached_reparses_when_zim_file_changes(tmp_path):
    zim_path = tmp_path / "fake.zim"
    zim_path.write_bytes(b"fake")
    cache_path = tmp_path / "cache.pkl"

    with patch(
        "wikipod.analysis.reader.read_articles_metadata_parallel", return_value=[]
    ) as mock_parse:
        read_articles_metadata_cached(zim_path, cache_path)
        zim_path.write_bytes(b"different content, different size")
        read_articles_metadata_cached(zim_path, cache_path)

    assert mock_parse.call_count == 2


def test_read_articles_metadata_cached_detects_nonexistent_file_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_articles_metadata_cached("does_not_exist.zim", tmp_path / "cache.pkl")


def test_read_articles_metadata_parallel_detects_nonexistent_file_path():
    with pytest.raises(FileNotFoundError):
        read_articles_metadata_parallel("does_not_exist.zim")


def test_read_articles_metadata_parallel_detects_wrong_file_type():
    with pytest.raises(ValueError):
        read_articles_metadata_parallel(__file__)


@requires_zim
def test_read_articles_metadata_parallel_matches_sequential_result():
    sequential = [extract_metadata(a) for a in iter_articles(ZIM_FILE)]

    parallel = read_articles_metadata_parallel(ZIM_FILE, workers=2)

    assert len(parallel) == len(sequential)
    assert {a.article_id for a in parallel} == {a.article_id for a in sequential}


@requires_zim
def test_read_articles_metadata_parallel_returns_article_metadata_instances():
    articles = read_articles_metadata_parallel(ZIM_FILE, workers=2)

    assert len(articles) > 0
    first = articles[0]
    assert isinstance(first.title, str) and first.title
    assert first.word_count >= 0
