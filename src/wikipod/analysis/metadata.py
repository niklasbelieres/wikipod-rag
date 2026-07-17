"""Derives structured `ArticleMetadata` from a raw `Article`."""

from wikipod.analysis.html_utils import (
    extract_categories,
    extract_links,
    get_content_root,
    split_into_sections,
)
from wikipod.analysis.models import Article, ArticleMetadata, Section


def extract_metadata(article: Article) -> ArticleMetadata:
    root = get_content_root(article.html)

    links = extract_links(root)
    sections = _build_sections(article, root)
    word_count = sum(len(section.text.split()) for section in sections)

    return ArticleMetadata(
        article_id=article.article_id,
        title=article.title,
        html_size_bytes=len(article.html.encode("utf-8")),
        word_count=word_count,
        link_count=len(links),
        links=links,
        section_count=len(sections),
        sections=sections,
        categories=extract_categories(article.html),
    )


def _build_sections(article: Article, root) -> list[Section]:
    return [
        Section(
            article_id=article.article_id,
            article_title=article.title,
            section_title=title,
            text=text,
        )
        for title, text in split_into_sections(root)
    ]
