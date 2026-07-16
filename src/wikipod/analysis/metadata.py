from bs4 import BeautifulSoup
import re

DROP_SELECTORS = [
    "style",
    "script",
    ".sidebar",
    ".navbox",
    ".infobox",
    ".metadata",
    ".mw-editsection",
    ".reference",
    ".reflist",
    ".zim-footer",
    ".toc",
]


def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for selector in DROP_SELECTORS:
        for el in soup.select(selector):
            el.decompose()
    return soup


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

from .models import Article, ArticleMetadata, Section

def extract_metadata(article: Article) -> ArticleMetadata:

    links = _extract_links(article.html)
    sections = _extract_sections(article)

    return ArticleMetadata(
        article_id=article.article_id,
        title=article.title,

        word_count=_count_words(
            article.html
        ),

        link_count=len(links),
        links=links,

        section_count=len(sections),
        sections=sections,

        categories=_extract_categories(
            article.html
        ),
    )
    

def _count_words(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    _clean_soup(soup)
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        return 0
    text = _normalize_text(content.get_text(" ", strip=True))
    return len(text.split())

def _extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    _clean_soup(soup)
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        return []
    links = []
    for link in content.find_all("a", href=True):
        href = link["href"]
        if href.startswith("http"):
            continue
        if href.startswith("./"):
            continue
        links.append(href)
    return links
    
def _extract_sections(article: Article) -> list[Section]:
    soup = BeautifulSoup(article.html, "html.parser")
    _clean_soup(soup)
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        return []
    sections = []
    current_title = "Lead"
    buffer = []
    def flush():
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                sections.append(Section(article_id=article.article_id ,article_title=article.title, section_title=current_title, text=text))
    for child in content.children:
        if getattr(child, "name", None) in ("h2", "h3"):
            flush()
            buffer.clear()
            heading_text = _normalize_text(child.get_text(" ", strip=True))
            current_title = heading_text if heading_text else "Untitled"
        elif getattr(child, "name", None) == "p":
            para_text = _normalize_text(child.get_text(" ", strip=True))
            if para_text:
                buffer.append(para_text)
    flush()
    return sections

def _count_sections(article: Article) -> int:
    return len(_extract_sections(article))

def _count_links(html: str) -> int:
    return len(_extract_links(html))
 
def _extract_categories(html: str) -> list[str]:
    return []
    