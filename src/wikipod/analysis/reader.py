"""
Reads articles out of a KIWIX .zim archive.
"""

import logging
import multiprocessing
import os
import pickle
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, wait
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


def _extract_one(archive: Archive, article_id: int, include_sections: bool) -> ArticleMetadata | None:
    """Extract metadata for one article_id, or None if it's a redirect/not an article."""
    entry = archive._get_entry_by_id(article_id)

    if entry.is_redirect:
        return None

    item = entry.get_item()
    html = item.content.tobytes().decode("utf-8", errors="ignore")

    if is_html_redirect(html):
        return None

    article = Article(article_id=article_id, title=entry.title, html=html)
    return extract_metadata(article, include_sections=include_sections)


def _extract_metadata_range(
    zim_path: str,
    start: int,
    end: int,
    counter=None,
    lock=None,
    report_every: int = 50,
    include_sections: bool = True,
) -> list[ArticleMetadata]:
    """Worker target: extract metadata for article ids in [start, end).

    Opens its own `Archive` handle rather than sharing one across processes --
    libzim's `Archive` wraps a C-extension object and isn't picklable, so each
    worker needs its own. ZIM files are read-only, so multiple independent
    handles on the same file are safe.

    `counter`/`lock` (proxies from a `multiprocessing.Manager`, *not* plain
    `multiprocessing.Value`/`Lock` -- those can only be inherited via fork,
    not passed through `ProcessPoolExecutor.submit()`, which breaks under the
    `spawn` start method macOS/Windows default to) are optional; when given,
    progress is reported in batches of `report_every` articles rather than
    after every single one, since each update is an IPC round-trip to the
    manager process -- doing that per-article would add real overhead across
    a multi-million-article corpus.
    """
    archive = Archive(zim_path)
    results: list[ArticleMetadata] = []
    pending_count = 0

    for article_id in range(start, end):
        try:
            result = _extract_one(archive, article_id, include_sections)
            if result is not None:
                results.append(result)
        except Exception:
            logger.warning("Skipping article %s", article_id, exc_info=True)
        finally:
            pending_count += 1
            if counter is not None and pending_count >= report_every:
                with lock:
                    counter.value += pending_count
                pending_count = 0

    if counter is not None and pending_count:
        with lock:
            counter.value += pending_count

    return results


def _extract_metadata_for_ids(
    zim_path: str, article_ids: list[int]
) -> list[ArticleMetadata]:
    """Worker target: full extraction (with section text) for specific article_ids."""
    archive = Archive(zim_path)
    results: list[ArticleMetadata] = []

    for article_id in article_ids:
        try:
            result = _extract_one(archive, article_id, include_sections=True)
            if result is not None:
                results.append(result)
        except Exception:
            logger.warning("Skipping article %s", article_id, exc_info=True)

    return results


METADATA_BATCH_SIZE = 5000


def read_articles_metadata_parallel(
    zim_path: str | Path,
    workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    batch_size: int = METADATA_BATCH_SIZE,
    include_sections: bool = True,
) -> list[ArticleMetadata]:
    """Read every non-redirect article and extract its metadata, in parallel.

    Same skip-and-log behavior as `iter_articles` + `extract_metadata`
    combined, just split across `workers` processes -- HTML parsing is
    CPU-bound pure Python, so `ProcessPoolExecutor` (real parallelism) is
    used instead of threads (which the GIL would block from helping here).

    Work is split into many small `batch_size`-sized chunks (not just
    `workers` equal-sized ones) so a `ProcessPoolExecutor` worker only ever
    holds one batch's `ArticleMetadata` objects in memory at a time --
    `ProcessPoolExecutor` automatically hands out the next batch as each
    worker finishes one. With only `workers` large ranges instead, each
    worker holds its *entire* multi-million-article share in memory at once.

    `include_sections=False` is the important one for a full-corpus pass over
    millions of articles: it drops each article's full body text (see
    `analysis.metadata.extract_metadata`), which is what actually exhausts
    RAM+swap on a 16 GB Pi once merged into one `articles` list -- word/link
    counts survive, only the text does not. Re-fetch full text for just the
    selected subset afterwards with `read_articles_metadata_for_ids`.

    `on_progress`, if given, is called as `on_progress(articles_done, total)`
    roughly once a second while workers are running -- this module has no
    opinion on how that's displayed (no `rich`/UI dependency here), that's
    up to the caller (see `cli.py`, which drives a progress bar off it).
    """
    zim_path = Path(zim_path)
    _validate_zim_path(zim_path)

    workers = workers or os.cpu_count() or 1
    total = Archive(str(zim_path)).article_count
    ranges = [(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]

    articles: list[ArticleMetadata] = []
    with multiprocessing.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(
                    _extract_metadata_range,
                    str(zim_path),
                    start,
                    end,
                    counter,
                    lock,
                    include_sections=include_sections,
                )
                for start, end in ranges
            }
            while pending:
                done, pending = wait(pending, timeout=1.0)
                for future in done:
                    articles.extend(future.result())
                if on_progress is not None:
                    on_progress(counter.value, total)

    if on_progress is not None:
        on_progress(total, total)

    return articles


def read_articles_metadata_for_ids(
    zim_path: str | Path,
    article_ids: list[int],
    workers: int | None = None,
    batch_size: int = METADATA_BATCH_SIZE,
) -> list[ArticleMetadata]:
    """Full extraction (with section text) for a specific, known set of article_ids.

    Meant to run *after* selection, on `result.selected`'s article_ids -- the
    initial full-corpus pass uses `include_sections=False` to stay within
    memory on the full corpus, so the selected subset needs its full text
    fetched separately before it can be chunked. Only touches the given IDs
    directly (`archive._get_entry_by_id`), not the rest of the corpus.
    """
    zim_path = Path(zim_path)
    _validate_zim_path(zim_path)

    workers = workers or os.cpu_count() or 1
    batches = [article_ids[i : i + batch_size] for i in range(0, len(article_ids), batch_size)]

    articles: list[ArticleMetadata] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_metadata_for_ids, str(zim_path), batch) for batch in batches]
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
    on_progress: Callable[[int, int], None] | None = None,
    include_sections: bool = True,
) -> list[ArticleMetadata]:
    """Like `read_articles_metadata_parallel`, but skips re-parsing the ZIM if a
    cache from a previous run of the *same* file is still valid.

    Re-parsing a multi-million-article ZIM is the expensive part of `wikipod
    index`; this exists so iterating on selection weights/storage budget
    doesn't force a full re-parse every time. The cache is keyed on the ZIM
    file's size and mtime, *and* `include_sections` -- a cache written with
    full section text is not a valid substitute for a lightweight request
    (wastes memory pointlessly) and, more importantly, a lightweight cache is
    not a valid substitute for a full request (would silently return articles
    with no body text where text was expected).
    """
    zim_path = Path(zim_path)
    cache_path = Path(cache_path)
    _validate_zim_path(zim_path)

    stat = zim_path.stat()
    cached = _load_cache(cache_path)
    if (
        cached is not None
        and cached["zim_size"] == stat.st_size
        and cached["zim_mtime"] == stat.st_mtime
        and cached.get("include_sections") == include_sections
    ):
        logger.info(
            "Using cached article metadata from %s (%d articles)", cache_path, len(cached["articles"])
        )
        return cached["articles"]

    articles = read_articles_metadata_parallel(
        zim_path, workers=workers, on_progress=on_progress, include_sections=include_sections
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(
            {
                "zim_size": stat.st_size,
                "zim_mtime": stat.st_mtime,
                "include_sections": include_sections,
                "articles": articles,
            },
            fh,
        )

    return articles
