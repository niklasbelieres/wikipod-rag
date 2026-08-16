"""
Reads articles out of a KIWIX .zim archive.
"""

import logging
import os
import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from libzim.reader import Archive

from wikipod.analysis.metadata import extract_metadata
from wikipod.analysis.models import Article, ArticleMetadata

logger = logging.getLogger(__name__)


def _validate_zim_path(zim_path: Path) -> None:
    if not zim_path.exists():
        raise FileNotFoundError(f"File {zim_path} does not exist")

    if zim_path.suffix != ".zim":
        raise ValueError(f"File {zim_path} is not a .zim file")


def iter_articles(zim_path: str | Path) -> Iterator[Article]:
    """Yield every non-redirect article contained in a .zim archive.

    Raises:
        FileNotFoundError: if ``zim_path`` does not exist.
        ValueError: if ``zim_path`` does not have a ``.zim`` extension.
    """
    zim_path = Path(zim_path)
    _validate_zim_path(zim_path)

    archive = Archive(str(zim_path))

    for article_id in range(archive.article_count):
        try:
            entry = archive._get_entry_by_id(article_id)

            if entry.is_redirect:
                continue

            item = entry.get_item()
            html = item.content.tobytes().decode("utf-8", errors="ignore")

            if is_html_redirect(html):
                continue

            yield Article(article_id=article_id, title=entry.title, html=html)
        except Exception:
            logger.warning("Skipping article %s", article_id, exc_info=True)


def is_html_redirect(html: str) -> bool:
    """Detect meta-refresh redirect pages that libzim doesn't flag as redirects itself."""
    return 'http-equiv="refresh"' in html and "URL=" in html


def _extract_metadata_range(zim_path: str, start: int, end: int) -> list[ArticleMetadata]:
    """Worker target: extract metadata for article ids in [start, end).

    Opens its own `Archive` handle rather than sharing one across processes --
    libzim's `Archive` wraps a C-extension object and isn't picklable, so each
    worker needs its own. ZIM files are read-only, so multiple independent
    handles on the same file are safe.
    """
    archive = Archive(zim_path)
    results: list[ArticleMetadata] = []

    for article_id in range(start, end):
        try:
            entry = archive._get_entry_by_id(article_id)

            if entry.is_redirect:
                continue

            item = entry.get_item()
            html = item.content.tobytes().decode("utf-8", errors="ignore")

            if is_html_redirect(html):
                continue

            article = Article(article_id=article_id, title=entry.title, html=html)
            results.append(extract_metadata(article))
        except Exception:
            logger.warning("Skipping article %s", article_id, exc_info=True)

    return results


def read_articles_metadata_parallel(
    zim_path: str | Path, workers: int | None = None
) -> list[ArticleMetadata]:
    """Read every non-redirect article and extract its metadata, in parallel.

    Same skip-and-log behavior as `iter_articles` + `extract_metadata`
    combined, just split across `workers` processes -- HTML parsing is
    CPU-bound pure Python, so `ProcessPoolExecutor` (real parallelism) is
    used instead of threads (which the GIL would block from helping here).
    """
    zim_path = Path(zim_path)
    _validate_zim_path(zim_path)

    workers = workers or os.cpu_count() or 1
    total = Archive(str(zim_path)).article_count
    step = max((total + workers - 1) // workers, 1)
    ranges = [(i, min(i + step, total)) for i in range(0, total, step)]

    articles: list[ArticleMetadata] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_extract_metadata_range, str(zim_path), start, end)
            for start, end in ranges
        ]
        for future in futures:
            articles.extend(future.result())

    return articles


def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        logger.warning("Failed to load article cache at %s, will re-parse.", cache_path, exc_info=True)
        return None


def read_articles_metadata_cached(
    zim_path: str | Path,
    cache_path: str | Path,
    workers: int | None = None,
) -> list[ArticleMetadata]:
    """Like `read_articles_metadata_parallel`, but skips re-parsing the ZIM if a
    cache from a previous run of the *same* file is still valid.

    Re-parsing a multi-million-article ZIM is the expensive part of `wikipod
    index`; this exists so iterating on selection weights/storage budget
    doesn't force a full re-parse every time. The cache is keyed on the ZIM
    file's size and mtime -- if either changed (different dump, or the same
    path re-downloaded), it's treated as stale and rebuilt automatically.
    """
    zim_path = Path(zim_path)
    cache_path = Path(cache_path)
    _validate_zim_path(zim_path)

    stat = zim_path.stat()
    cached = _load_cache(cache_path)
    if cached is not None and cached["zim_size"] == stat.st_size and cached["zim_mtime"] == stat.st_mtime:
        logger.info(
            "Using cached article metadata from %s (%d articles)", cache_path, len(cached["articles"])
        )
        return cached["articles"]

    articles = read_articles_metadata_parallel(zim_path, workers=workers)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(
            {"zim_size": stat.st_size, "zim_mtime": stat.st_mtime, "articles": articles}, fh
        )

    return articles
