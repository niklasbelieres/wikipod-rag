"""Splits article sections into fixed-size, overlapping word-chunks.

Chunking happens per-section (rather than per-article) so that a chunk never
mixes content from two unrelated headings, which keeps retrieval results
coherent.
"""

from wikipod.analysis.models import ArticleMetadata, Section
from wikipod.chunking.models import Chunk
from wikipod.config import ChunkingConfig


def chunk_section(
    section: Section, max_words: int = 250, overlap: int = 40, *, section_index: int = 0
) -> list[Chunk]:
    """Split a single section into overlapping chunks of at most `max_words` words.

    `section_index` is the zero-based position in the article. Callers processing
    multiple sections must provide it; `chunk_article` assigns it automatically.

    Raises:
        ValueError: if max_words <= 0 or overlap is outside [0, max_words).
    """
    settings = ChunkingConfig(max_words=max_words, overlap=overlap)
    max_words, overlap = settings.max_words, settings.overlap

    words = section.text.split()

    if len(words) <= max_words:
        return [_make_chunk(section, section_index=section_index, chunk_index=0, words=words)]

    step = max_words - overlap
    chunks = []
    chunk_index = 0

    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        chunks.append(
            _make_chunk(
                section, section_index=section_index, chunk_index=chunk_index, words=window
            )
        )
        chunk_index += 1
        if start + max_words >= len(words):
            break

    return chunks


def chunk_article(
    article: ArticleMetadata, max_words: int = 250, overlap: int = 40
) -> list[Chunk]:
    """Chunk every section of an article and return a flat list of chunks."""
    settings = ChunkingConfig(max_words=max_words, overlap=overlap)
    max_words, overlap = settings.max_words, settings.overlap
    chunks: list[Chunk] = []
    for section_index, section in enumerate(article.sections):
        chunks.extend(
            chunk_section(
                section, max_words=max_words, overlap=overlap, section_index=section_index
            )
        )
    return chunks


def _make_chunk(
    section: Section, section_index: int, chunk_index: int, words: list[str]
) -> Chunk:
    return Chunk(
        article_id=section.article_id,
        article_title=section.article_title,
        section_title=section.section_title,
        section_index=section_index,
        chunk_index=chunk_index,
        word_count=len(words),
        text=" ".join(words),
    )
