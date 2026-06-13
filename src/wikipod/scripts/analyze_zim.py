from wikipod.analysis.reader import iter_articles
from wikipod.analysis.metadata import extract_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ZIM_FILE = (
    PROJECT_ROOT
    / "test"
    / "data"
    / "climate-change-mini.zim"
)
all_metadata = []

for article in iter_articles(zim_path=ZIM_FILE):
    metadata = extract_metadata(article)
    
    all_metadata.append(metadata)
    
top_articles = sorted(
    all_metadata,
    key=lambda m: m.link_count,
    reverse=True,
)

print(f"Articles: {len(all_metadata)}")

print(
    f"Avg words: "
    f"{sum(m.word_count for m in all_metadata) / len(all_metadata):.2f}"
)

print(
    f"Avg links: "
    f"{sum(m.link_count for m in all_metadata) / len(all_metadata):.2f}"
)