"""
Core domain models shared across the pipeline.
"""
from pydantic import BaseModel, Field

class Article(BaseModel):
    """A single raw article as read from a ZIM archive."""

    article_id: int
    title: str
    html: str


class Section(BaseModel):
    """A titled block of text within an article (e.g. an H2/H3 section)."""

    article_id: int
    article_title: str
    section_title: str
    text: str


class ArticleMetadata(BaseModel):
    """Derived, structured information about an article, used for both
    selection scoring and downstream chunking."""

    article_id: int
    title: str

    html_size_bytes: int = Field(description="Size of the raw HTML, used as a storage proxy.")
    word_count: int

    link_count: int
    links: list[str]

    section_count: int
    sections: list[Section]

    categories: list[str]
