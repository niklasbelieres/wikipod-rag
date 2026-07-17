from pathlib import Path

import pytest

from wikipod.analysis.html_utils import (
    extract_categories,
    extract_links,
    get_content_root,
    split_into_sections,
)
from wikipod.analysis.metadata import extract_metadata
from wikipod.analysis.models import Article, ArticleMetadata
from wikipod.analysis.reader import iter_articles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIM_FILE = PROJECT_ROOT / "test" / "data" / "climate-change-mini.zim"


# -- html_utils --------------------------------------------------------------


def test_extract_links_returns_internal_links_only():
    html = """
    <div class="mw-parser-output">
        <a href="Article_A">A</a>
        <a href="https://example.com">external</a>
        <a href="./Article_B">also skipped</a>
        <a href="Article_C">C</a>
    </div>
    """
    root = get_content_root(html)
    assert extract_links(root) == ["Article_A", "Article_C"]


def test_extract_links_returns_empty_list_when_no_links_exist():
    html = '<div class="mw-parser-output"><p>No links here</p></div>'
    assert extract_links(get_content_root(html)) == []


def test_split_into_sections_returns_h2_and_h3_titles():
    html = """
    <div class="mw-parser-output">
        <h2>History</h2><p>...</p>
        <h3>Europe</h3><p>...</p>
        <h2>Future</h2><p>...</p>
    </div>
    """
    titles = [title for title, _ in split_into_sections(get_content_root(html))]
    assert titles == ["History", "Europe", "Future"]


def test_split_into_sections_groups_leading_text_as_lead():
    html = '<div class="mw-parser-output"><p>Only text</p></div>'
    sections = split_into_sections(get_content_root(html))
    assert sections == [("Lead", "Only text")]


def test_extract_categories_reads_catlinks_box():
    html = """
    <html><body>
      <div class="mw-parser-output"><p>Body</p></div>
      <div id="catlinks">
        <a href="../Category:Climate_change">Climate change</a>
        <a href="../Category:Physics">Physics</a>
      </div>
    </body></html>
    """
    assert extract_categories(html) == ["Climate change", "Physics"]


def test_extract_categories_returns_empty_list_when_absent():
    html = '<div class="mw-parser-output"><p>No categories</p></div>'
    assert extract_categories(html) == []


# -- metadata.extract_metadata (in-memory, no ZIM required) ------------------


def test_extract_metadata_from_synthetic_article():
    html = """
    <html><body>
      <div class="mw-parser-output">
        <p>Climate change refers to long term shifts.</p>
        <h2>Causes</h2>
        <p>Human activity is the main driver.</p>
        <a href="Greenhouse_gas">link</a>
      </div>
      <div id="catlinks"><a href="../Category:Climate">Climate</a></div>
    </body></html>
    """
    article = Article(article_id=1, title="Climate change", html=html)
    metadata = extract_metadata(article)

    assert isinstance(metadata, ArticleMetadata)
    assert metadata.article_id == 1
    assert metadata.title == "Climate change"
    assert metadata.word_count > 0
    assert metadata.links == ["Greenhouse_gas"]
    assert metadata.link_count == 1
    assert metadata.section_count == 2  # Lead + Causes
    assert metadata.categories == ["Climate"]
    assert metadata.html_size_bytes == len(html.encode("utf-8"))


def test_extract_metadata_handles_missing_content_root():
    article = Article(article_id=2, title="Empty", html="<html><body></body></html>")
    metadata = extract_metadata(article)

    assert metadata.word_count == 0
    assert metadata.links == []
    assert metadata.sections == []


# -- integration against the real demo ZIM, if present -----------------------


@pytest.mark.skipif(
    not ZIM_FILE.exists(), reason="demo ZIM file not present; see test/data/README.md"
)
def test_extract_metadata_on_real_article_is_internally_consistent():
    article = next(iter_articles(ZIM_FILE))
    metadata = extract_metadata(article)

    assert metadata.article_id == article.article_id
    assert metadata.title == article.title
    assert metadata.link_count == len(metadata.links)
    assert metadata.section_count == len(metadata.sections)
