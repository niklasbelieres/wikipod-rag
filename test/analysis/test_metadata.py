from pathlib import Path

from wikipod.analysis.metadata import (
    extract_metadata,
    _count_words,
    _extract_links,
    _count_links,
    _extract_sections,
    _count_sections
)
from wikipod.analysis.reader import (
    iter_articles
)

from wikipod.analysis.models import (
    ArticleMetadata
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIM_FILE = (
    PROJECT_ROOT
    / "test"
    / "data"
    / "climate-change-mini.zim"
)

def test_count_words_returns_positive_number():
    html = """
    <div class="mw-parser-output">
        <p>Hello world</p>
        <p>This is a test</p>
    </div>
    """
    
    assert _count_words(html) == 6
    

def test_count_words_returns_zero_when_content_div_missing():
    html = """
    <html>
        <body>
            <p>Hello world</p>
        </body>
    </html>
    """
    
    assert _count_words(html) == 0

def test_extract_links_returns_all_links():
    html = """
    <div class="mw-parser-output">
        <a href="Article_A">A</a>
        <a href="Article_B">B</a>
        <a href="Article_C">C</a>
    </div>
    """
    
    links = _extract_links(html)

    assert links == [
        "Article_A",
        "Article_B",
        "Article_C",
    ]

def test_extract_links_returns_empty_list_when_no_links_exist():
    html = """
    <div class="mw-parser-output">
        <p>No links here</p>
    </div>
    """
    
    links = _extract_links(html)
    
    assert links == []

def test_count_links_matches_number_of_extracted_links():
    html = """
    <div class="mw-parser-output">
        <a href="A">A</a>
        <a href="B">B</a>
        <a href="C">C</a>
    </div>
    """
    
    links = _extract_links(html)
    assert _count_links(html) == len(links)

def test_extract_sections_returns_h2_and_h3_titles():
    html = """
    <div class="mw-parser-output">
        <h2>History</h2>
        <p>...</p>

        <h3>Europe</h3>
        <p>...</p>

        <h2>Future</h2>
    </div>
    """
    
    sections = _extract_sections(html)
    
    assert sections == [
        "History",
        "Europe",
        "Future",
    ]
    

def test_extract_sections_returns_empty_list_when_no_sections_exist():
    html = """
    <div class="mw-parser-output">
        <p>Only text</p>
    </div>
    """
    
    assert _extract_sections(html) == []

def test_count_sections_matches_number_of_extracted_sections():
    html = """
    <div class="mw-parser-output">
        <h2>History</h2>
        <h2>Future</h2>
        <h3>Europe</h3>
    </div>
    """
    
    sections = _extract_sections(html)
    
    assert len(sections) == _count_sections(html)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIM_FILE = (
    PROJECT_ROOT
    / "test"
    / "data"
    / "climate-change-mini.zim"
)

def test_extract_metadata_preserves_article_id():
    article = next(iter_articles(ZIM_FILE))
    
    metadata = extract_metadata(article)
    
    assert metadata.article_id == article.article_id
    assert metadata.title == article.title

def test_extract_metadata_preserves_title():
    article = next(iter_articles(ZIM_FILE))

    metadata = extract_metadata(article)

    assert metadata.title == article.title

def test_extract_metadata_populates_word_count():
    article = next(iter_articles(ZIM_FILE))

    metadata = extract_metadata(article)

    assert metadata.word_count > 0

def test_extract_metadata_populates_links():
    article = next(iter_articles(ZIM_FILE))

    metadata = extract_metadata(article)

    assert metadata.links == _extract_links(article.html)
    assert metadata.link_count == len(metadata.links)

def test_extract_metadata_populates_section_count():
    article = next(iter_articles(ZIM_FILE))

    metadata = extract_metadata(article)

    assert metadata.sections == _extract_sections(article.html)
    assert metadata.section_count == len(metadata.sections)

def test_extract_metadata_returns_article_metadata_instance():
    article = next(iter_articles(ZIM_FILE))

    metadata = extract_metadata(article)

    assert isinstance(metadata, ArticleMetadata)