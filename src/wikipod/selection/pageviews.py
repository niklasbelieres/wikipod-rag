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
    from wikipod.selection.pageviews import load_pageviews

    # 1. Download and decompress a dump, e.g.:
    #    curl -O https://dumps.wikimedia.org/other/pageviews/2026/2026-06/pageviews-20260601-000000.gz
    #    gunzip pageviews-20260601-000000.gz
    pageviews = load_pageviews("pageviews-20260601-000000")

    # 2. Pass it into scoring wherever an article title needs a view count.
    #    Always go through get_views() rather than a raw dict lookup --
    #    it normalizes title formatting differences (see below).
    from wikipod.selection.pageviews import get_views
    views = get_views(pageviews, "Climate change")

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
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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

    views: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
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
