from bs4 import BeautifulSoup

from .models import  (Article, ArticleMetadata)

def extract_metadata(article: Article) -> ArticleMetadata:

    links = _extract_links(article.html)
    sections = _extract_sections(article.html)

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

    content = soup.find(
        "div",
        class_="mw-parser-output"
    )

    if content is None:
        return 0

    text = content.get_text(
        " ",
        strip=True
    )

    return len(text.split())

def _extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    if soup is None:
        return []

    links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if href.startswith("http"):
            continue

        if href.startswith("./"):
            continue

        links.append(href)
    return links
    
def _extract_sections(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    return [
        section.get_text(strip=True)
        for section in soup.find_all(["h2", "h3"])
    ]

def _count_sections(html: str) -> int:
    return len(_extract_sections(html))

def _count_links(html: str) -> int:
    return len(_extract_links(html))
 
def _extract_categories(html: str) -> list[str]:
    return []
    