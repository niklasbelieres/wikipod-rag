from pathlib import Path
import pytest

from wikipod.analysis.reader import (
    iter_articles,
    is_html_redirect,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIM_FILE = (
    PROJECT_ROOT
    / "test"
    / "data"
    / "climate-change-mini.zim"
)

def test_detects_non_existend_file_path():
    with pytest.raises(FileNotFoundError):
        next(iter_articles("non_existend_file.zim"))

def test_detects_wrong_file_type():
    with pytest.raises(ValueError):
        next(iter_articles("main.py"))

def test_detects_html_redirect():
    html = """
    <html>
      <head>
        <meta http-equiv="refresh"
              content="0;URL='./Target'" />
      </head>
    </html>
    """

    assert is_html_redirect(html) is True


def test_detects_normal_article():
    html = """
    <html>
      <body>
        <h1>Climate change</h1>
      </body>
    </html>
    """

    assert is_html_redirect(html) is False

def test_reader_returns_articles():
    articles = []

    for article in iter_articles(str(ZIM_FILE)):
        articles.append(article)

        if len(articles) == 5:
            break

    assert len(articles) > 0


def test_articles_have_titles():
    article = next(iter_articles(str(ZIM_FILE)))

    assert article.title
    assert isinstance(article.title, str)


def test_articles_have_html():
    article = next(iter_articles(str(ZIM_FILE)))

    assert article.html
    assert isinstance(article.html, str)


def test_articles_contain_html_document():
    article = next(iter_articles(str(ZIM_FILE)))

    assert "<html" in article.html.lower()


def test_reader_skips_html_redirects():
    for article in iter_articles(str(ZIM_FILE)):
        assert not is_html_redirect(article.html)


def test_reader_returns_unique_article_ids():
    ids = []

    for article in iter_articles(str(ZIM_FILE)):
        ids.append(article.article_id)

        if len(ids) >= 100:
            break

    assert len(ids) == len(set(ids))


def test_reader_returns_multiple_articles():
    count = 0

    for _ in iter_articles(str(ZIM_FILE)):
        count += 1

        if count >= 20:
            break

    assert count >= 20