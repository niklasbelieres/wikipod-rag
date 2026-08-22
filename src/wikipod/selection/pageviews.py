"""Loads Wikipedia pageview counts as an optional article-scoring signal.

Wikimedia publishes hourly/daily/monthly pageview dumps at
https://dumps.wikimedia.org/other/pageviews/ as gzip-compressed,
whitespace-separated text, one row per page:

    <domain_code> <page_title> <view_count> <byte_size>

e.g.:

    en Climate_change 4821 0
    en.m Climate_change 9110 0

Usage
-----
Either bring your own already-downloaded dump:

    from wikipod.selection.pageviews import load_pageviews, get_views

    # 1. Download and decompress a dump, e.g.:
    #    curl -O https://dumps.wikimedia.org/other/pageviews/2026/2026-06/pageviews-20260601-000000.gz
    #    gunzip pageviews-20260601-000000.gz
    pageviews = load_pageviews("pageviews-20260601-000000")
    views = get_views(pageviews, "Climate change")

...or fetch and aggregate a full day directly (single hours are noisy --
most articles get few or zero views in any given hour):

    from datetime import date
    from wikipod.selection.pageviews import download_pageviews_day, save_pageviews

    pageviews = download_pageviews_day(date(2026, 6, 1))
    save_pageviews(pageviews, "pageviews-20260601-aggregated.txt")
    # now point config.paths.pageviews_file at that file -- load_pageviews()
    # reads it back exactly like any other dump, no other code needs to change.

Notes
-----
- `domain_prefix` filters to a single domain code (default "en" for
  desktop en.wikipedia.org, as opposed to "en.m" for mobile); pass a
  different code, or aggregate several dumps, as needed.
- Source titles use underscores ("Climate_change"); titles from other parts
  of a pipeline (e.g. a ZIM/MediaWiki reader) often use spaces
  ("Climate change"). `get_views()` normalizes both to the same form so
  lookups don't silently miss -- use it instead of indexing the dict
  directly.
- If no pageview data is available for a given deployment, treat this
  signal as optional: fall back to another popularity proxy (e.g. an
  in-corpus link-frequency count) or omit it from scoring entirely.
- `download_pageviews_hour`/`download_pageviews_day` process one hour at a
  time entirely in memory (download -> gunzip -> filter -> discard) rather
  than writing 24 raw multi-GB dumps to disk at once -- relevant on
  constrained hardware like the Pi.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PAGEVIEWS_BASE_URL = "https://dumps.wikimedia.org/other/pageviews"

# Wikimedia rejects requests without a descriptive User-Agent (their anti-abuse
# policy, see https://meta.wikimedia.org/wiki/User-Agent_policy) -- requests'
# default UA gets a 403, not the 404 you'd expect for a missing dump.
_REQUEST_HEADERS = {"User-Agent": "wikipod-rag/0.1 (student project, htw saar; RAG on Raspberry Pi)"}


def _parse_pageview_lines(lines: Iterable[str], domain_prefix: str) -> dict[str, int]:
    views: dict[str, int] = {}
    for line in lines:
        parts = line.split(" ")
        if len(parts) < 3:
            continue
        domain, title, count = parts[0], parts[1], parts[2]
        if domain != domain_prefix:
            continue
        try:
            key = normalize_title(title)
            views[key] = views.get(key, 0) + int(count)
        except ValueError:
            continue
    return views


def load_pageviews(path: str | Path, domain_prefix: str = "en") -> dict[str, int]:
    """Parse a decompressed Wikimedia pageviews dump into {normalized_title: view_count}.

    Args:
        path: path to a plain-text pageviews dump (gunzip it first).
        domain_prefix: only keep rows for this domain code.

    Returns:
        An empty dict, with a warning logged, if `path` doesn't exist --
        callers can treat a missing file as "no pageview data available"
        rather than a fatal error.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Pageviews file %s not found; returning empty map.", path)
        return {}

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return _parse_pageview_lines(fh, domain_prefix)


def download_pageviews_hour(day: date, hour: int, domain_prefix: str = "en") -> dict[str, int]:
    """Download and parse one hourly Wikimedia pageviews dump, entirely in memory.

    Never writes the (multi-GB, all-languages) raw dump to disk -- downloads
    the gzip bytes, decompresses and filters to `domain_prefix` in memory,
    and discards everything except the small resulting {title: count} map.

    Returns an empty dict (with a warning logged) on any network/parse
    failure, so one bad hour doesn't abort a full-day aggregation -- same
    skip-and-log philosophy as `analysis/reader.py`'s article reading.
    """
    url = f"{PAGEVIEWS_BASE_URL}/{day:%Y}/{day:%Y-%m}/pageviews-{day:%Y%m%d}-{hour:02d}0000.gz"
    try:
        response = requests.get(url, timeout=60, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        decompressed = gzip.decompress(response.content).decode("utf-8", errors="ignore")
        return _parse_pageview_lines(decompressed.splitlines(), domain_prefix)
    except Exception:
        logger.warning("Failed to download/parse pageviews for %s hour %02d", day, hour, exc_info=True)
        return {}


def download_pageviews_day(
    day: date,
    domain_prefix: str = "en",
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Download and aggregate all 24 hourly dumps for `day` into one pageview map.

    A single hour is noisy -- most articles get few or zero views in any
    given hour -- summing a full day gives a much more stable popularity
    signal. `on_progress(hours_done, 24)`, if given, is called after each
    hour (see `cli.py`'s `fetch-pageviews` command for a progress-bar use).
    """
    merged: dict[str, int] = {}
    for hour in range(24):
        for key, count in download_pageviews_hour(day, hour, domain_prefix).items():
            merged[key] = merged.get(key, 0) + count
        if on_progress is not None:
            on_progress(hour + 1, 24)
    return merged


def save_pageviews(views: dict[str, int], path: str | Path, domain_prefix: str = "en") -> None:
    """Write an aggregated {title: count} map back out in the same plain-text
    format `load_pageviews` reads -- so the result of `download_pageviews_day`
    can be pointed at directly via `config.paths.pageviews_file` without any
    other part of the pipeline needing to change.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for title, count in views.items():
            fh.write(f"{domain_prefix} {title} {count} 0\n")


def normalize_title(title: str) -> str:
    """Normalize a title to MediaWiki's underscore convention for stable lookups.

    "Climate change" and "Climate_change" both map to "Climate_change".
    """
    return title.strip().replace(" ", "_")


def get_views(pageviews: dict[str, int], article_title: str) -> int:
    """Look up an article's view count by title, normalizing first.

    Prefer this over `pageviews.get(title)` / `pageviews[title]` directly --
    a raw lookup with an un-normalized title (e.g. containing spaces) can
    silently miss even when the data is present under its underscored form.
    """
    return pageviews.get(normalize_title(article_title), 0)
