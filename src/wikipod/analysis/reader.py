"""
Reads articles out of a KIWIX .zim archive.
"""

import json
import logging
import multiprocessing
import os
import pickle
import sys
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


def _extract_one(
    archive: Archive, article_id: int, include_sections: bool
) -> ArticleMetadata | None:
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
    """Worker target: extract metadata for article ids in [start, end)."""
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


def iter_articles_metadata_parallel(
    zim_path: str | Path,
    workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    batch_size: int = METADATA_BATCH_SIZE,
    include_sections: bool = True,
) -> Iterator[ArticleMetadata]:
    """Read every non-redirect article and extract its metadata, in parallel,
    yielding each one as it arrives rather than accumulating a list.

    Same skip-and-log behavior as `iter_articles` + `extract_metadata`
    combined, just split across `workers` processes.

    Work is split into many small `batch_size`-sized chunks (not just
    `workers` equal-sized ones) so a `ProcessPoolExecutor` worker only ever
    holds one batch's `ArticleMetadata` objects in memory at a time.
    `ProcessPoolExecutor` automatically hands out the next batch as each
    worker finishes one.

    `include_sections=False` for a full-corpus pass:
    it drops each article's full body text (see `analysis.metadata.extract_metadata`). 
    Re-fetch full text for just the selected subset afterwards with
    `read_articles_metadata_for_ids`.
    """
    zim_path = Path(zim_path)
    _validate_zim_path(zim_path)

    workers = workers or os.cpu_count() or 1
    total = Archive(str(zim_path)).article_count
    ranges = [(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]

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
                    for article in future.result():
                        _reintern_in_main_process(article)
                        yield article
                if on_progress is not None:
                    on_progress(counter.value, total)

    if on_progress is not None:
        on_progress(total, total)


def read_articles_metadata_parallel(
    zim_path: str | Path,
    workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    batch_size: int = METADATA_BATCH_SIZE,
    include_sections: bool = True,
) -> list[ArticleMetadata]:
    """`iter_articles_metadata_parallel`, materialized as a list.

    Works well for small and medium ZIM-files. For a full en-dump, the system
    ran out of memory.
    """
    return list(
        iter_articles_metadata_parallel(
            zim_path,
            workers=workers,
            on_progress=on_progress,
            batch_size=batch_size,
            include_sections=include_sections,
        )
    )


def _reintern_in_main_process(article: ArticleMetadata) -> None:
    """Re-intern `links`/`categories` after crossing a process boundary.

    `sys.intern()` inside a worker (see `analysis.html_utils`) only
    deduplicates strings within *that* worker's own interpreter -- results
    shipped back via `ProcessPoolExecutor`'s pickling get deserialized as
    fresh string objects in the main process, undoing the worker-local
    sharing. Re-interning here restores full-corpus-wide deduplication
    across *all* workers' contributions, not just within each one.
    """
    article.links = [sys.intern(link) for link in article.links]
    article.categories = [sys.intern(category) for category in article.categories]


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
        futures = [
            pool.submit(_extract_metadata_for_ids, str(zim_path), batch) for batch in batches
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
        logger.warning(
            "Failed to load article cache at %s, will re-parse.", cache_path, exc_info=True
        )
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
            "Using cached article metadata from %s (%d articles)",
            cache_path,
            len(cached["articles"]),
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


def _jsonl_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".meta.json")


def _jsonl_cache_is_valid(
    cache_path: Path, zim_stat: os.stat_result, include_sections: bool
) -> bool:
    meta_path = _jsonl_cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return (
        meta.get("zim_size") == zim_stat.st_size
        and meta.get("zim_mtime") == zim_stat.st_mtime
        and meta.get("include_sections") == include_sections
    )


def _iter_jsonl_articles(cache_path: Path) -> Iterator[ArticleMetadata]:
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ArticleMetadata.model_validate_json(line)


def stream_articles_metadata_cached(
    zim_path: str | Path,
    cache_path: str | Path,
    workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    include_sections: bool = True,
    batch_size: int = METADATA_BATCH_SIZE,
) -> Iterator[ArticleMetadata]:
    """JSONL-backed streaming cache: yields one `ArticleMetadata` at a time,
    never holding the full corpus in memory -- the full-corpus counterpart to
    `read_articles_metadata_cached` (which materializes a list, fine for
    smaller ZIMs but not for the full en.wikipedia corpus).

    On a cache hit (same ZIM size/mtime and `include_sections` as recorded in
    the `<cache_path>.meta.json` sidecar), streams straight from `cache_path`
    -- cheap, no HTML re-parsing. On a miss, runs the parallel extraction and
    writes each article to `cache_path` as it's yielded (write-through), so
    building `link_frequency_map` and then scoring -- both need a full pass,
    see `selection.selector.select_within_budget` -- means calling this
    twice, and the second call is always a cheap disk read regardless of
    whether the first call populated the cache or found it already valid.
    """
    zim_path = Path(zim_path)
    cache_path = Path(cache_path)
    _validate_zim_path(zim_path)

    stat = zim_path.stat()
    if _jsonl_cache_is_valid(cache_path, stat, include_sections):
        logger.info("Streaming cached article metadata from %s", cache_path)
        yield from _iter_jsonl_articles(cache_path)
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        for article in iter_articles_metadata_parallel(
            zim_path,
            workers=workers,
            on_progress=on_progress,
            batch_size=batch_size,
            include_sections=include_sections,
        ):
            fh.write(article.model_dump_json())
            fh.write("\n")
            yield article

    _jsonl_cache_meta_path(cache_path).write_text(
        json.dumps(
            {
                "zim_size": stat.st_size,
                "zim_mtime": stat.st_mtime,
                "include_sections": include_sections,
            }
        )
    )
