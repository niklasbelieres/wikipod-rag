"""Flat (non-sectioned) plain-text extraction.

Useful when you want an article's full text as a single blob (e.g. quick
inspection, or whole-article embeddings) rather than split into sections --
for section-aware chunking, use `chunking.chunker` on an `ArticleMetadata`
produced by `analysis.metadata.extract_metadata` instead.
"""
from wikipod.analysis.html_utils import clean_text, get_content_root


def extract_article_text(html: str) -> str:
    """Return the article's cleaned, flattened body text, or "" if it has no content root."""
    root = get_content_root(html)
    if root is None:
        return ""
    return clean_text(root.get_text(separator=" ", strip=True))
