"""Measures the real, uncompressed HTML size of the configured ZIM's corpus.

`storage_budget_mb` in config/*.yaml is checked against each article's
`html_size_bytes` (raw HTML, see selection/selector.py), not the ZIM file's
own (compressed) size on disk -- so the ZIM file size is only a weak proxy
for how large a budget makes sense. This script sums the real figure and
suggests a few budget options as a starting point.

Usage:
    WIKIPOD_ENV=server python -m wikipod.scripts.measure_corpus_size
"""

from __future__ import annotations

from wikipod.analysis.reader import stream_articles_metadata_cached
from wikipod.config import get_config

MB = 1024**2


def main() -> None:
    config = get_config()
    zim_path = config.resolve_path(config.paths.zim_file)
    cache_path = (
        config.resolve_path(config.paths.data_dir) / f"{zim_path.stem}_metadata_cache.jsonl"
    )

    print(f"Scanning {zim_path} ...")

    total_bytes = 0
    article_count = 0
    for article in stream_articles_metadata_cached(
        zim_path, cache_path, include_sections=False
    ):
        total_bytes += article.html_size_bytes
        article_count += 1
        if article_count % 5000 == 0:
            print(f"  ... {article_count} Artikel, {total_bytes / MB:.0f} MB bisher")

    total_mb = total_bytes / MB
    print(f"\n{article_count} Artikel, insgesamt {total_mb:.0f} MB unkomprimiertes HTML")
    print("\nMögliche Budgets (storage_budget_mb):")
    for fraction in (0.3, 0.4, 0.5):
        print(f"  {int(fraction * 100)}%: {total_mb * fraction:.0f} MB")


if __name__ == "__main__":
    main()
