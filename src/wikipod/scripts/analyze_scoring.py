"""Analyze the scale and weighted contribution of WikiPod selection scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wikipod.analysis.reader import stream_articles_metadata_cached
from wikipod.analysis.statistics import link_frequency_map
from wikipod.config import get_config
from wikipod.selection.pageviews import get_views, load_pageviews


@dataclass
class ScoreComponents:
    word_count: float
    link_count: float
    importance: float
    incoming_links: float
    pageviews: float


def calculate_components(
    article,
    link_frequencies: dict[str, int],
    pageviews: dict[str, int],
) -> ScoreComponents:
    """Return the five unweighted score components used by scoring.py."""

    word_score = math.log1p(article.word_count)
    link_score = math.log1p(article.link_count)

    importance_score = sum(
        math.log1p(link_frequencies.get(link, 0))
        for link in article.links
    )

    incoming_links_score = math.log1p(
        link_frequencies.get(article.title, 0)
    )

    pageviews_score = math.log1p(
        get_views(pageviews, article.title)
    )

    return ScoreComponents(
        word_count=word_score,
        link_count=link_score,
        importance=importance_score,
        incoming_links=incoming_links_score,
        pageviews=pageviews_score,
    )


def summarize(values: list[float]) -> dict[str, float]:
    """Return basic statistics for one score component."""

    array = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def main() -> None:
    config = get_config()

    zim_path = config.resolve_path(config.paths.zim_file)
    cache_path = (
        config.resolve_path(config.paths.data_dir) / f"{zim_path.stem}_metadata_cache.jsonl"
    )

    print(f"Reading articles from {zim_path}")

    # Same parallel + on-disk-cache path as `measure_corpus_size.py` and
    # `wikipod index` pass 1 (include_sections=False -- only word_count/
    # link_count/links/title are needed here, not full section text). Reuses
    # the cache if one of those already built it for this ZIM, and builds it
    # here otherwise so a later `wikipod index` run gets a free cache hit.
    articles = list(
        stream_articles_metadata_cached(zim_path, cache_path, include_sections=False)
    )

    print(f"Loaded {len(articles)} articles")

    link_frequencies = link_frequency_map(articles)

    pageviews: dict[str, int] = {}

    if config.paths.pageviews_file:
        pageviews_path = config.resolve_path(
            config.paths.pageviews_file
        )

        pageviews = load_pageviews(pageviews_path)

    raw: dict[str, list[float]] = {
        "word_count": [],
        "link_count": [],
        "importance": [],
        "incoming_links": [],
        "pageviews": [],
    }

    weighted: dict[str, list[float]] = {
        name: [] for name in raw
    }

    weights = config.selection.weights

    for article in articles:
        components = calculate_components(
            article,
            link_frequencies,
            pageviews,
        )

        for name, value in vars(components).items():
            raw[name].append(value)

            weight = getattr(weights, name)
            weighted[name].append(value * weight)

    print("\nRaw score components")
    print(
        f"{'component':<18} "
        f"{'mean':>10} "
        f"{'median':>10} "
        f"{'p95':>10} "
        f"{'max':>10}"
    )

    for name, values in raw.items():
        stats = summarize(values)

        print(
            f"{name:<18} "
            f"{stats['mean']:>10.3f} "
            f"{stats['median']:>10.3f} "
            f"{stats['p95']:>10.3f} "
            f"{stats['max']:>10.3f}"
        )

    print("\nWeighted score contributions")
    print(
        f"{'component':<18} "
        f"{'weight':>8} "
        f"{'mean':>10} "
        f"{'median':>10} "
        f"{'p95':>10} "
        f"{'max':>10}"
    )

    for name, values in weighted.items():
        stats = summarize(values)
        weight = getattr(weights, name)

        print(
            f"{name:<18} "
            f"{weight:>8.3f} "
            f"{stats['mean']:>10.3f} "
            f"{stats['median']:>10.3f} "
            f"{stats['p95']:>10.3f} "
            f"{stats['max']:>10.3f}"
        )

    total_mean = sum(
        summarize(values)["mean"]
        for values in weighted.values()
    )

    print("\nAverage contribution to total weighted score")

    for name, values in weighted.items():
        mean_contribution = summarize(values)["mean"]

        if total_mean:
            share = mean_contribution / total_mean * 100.0
        else:
            share = 0.0

        print(f"{name:<18} {share:>7.2f}%")


if __name__ == "__main__":
    main()