# from pathlib import Path
# import pytest

# from wikipod.analysis.reader import (
#     iter_articles,
#     is_html_redirect,
# )
# from wikipod.analysis.metadata import extract_metadata
# from wikipod.analysis.statistics import (
#     most_common_links,
#     summarize_articles
# )

# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ZIM_FILE = (
#     PROJECT_ROOT
#     / "test"
#     / "data"
#     / "climate-change-mini.zim"
# )

# all_meta = [
#     extract_metadata(a)
#     for a in iter_articles(ZIM_FILE)
# ]

# summary = summarize_articles(all_meta)

# print(summary)
# print()
# print(most_common_links(all_meta, 50))