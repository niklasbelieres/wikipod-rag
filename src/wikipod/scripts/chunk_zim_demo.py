from pathlib import Path

from wikipod.analysis.metadata import extract_metadata
from wikipod.analysis.reader import iter_articles
from wikipod.chunking.chunker import chunk_article

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ZIM_FILE = (
    PROJECT_ROOT
    / "test"
    / "data"
    / "climate-change-mini.zim"
)


def main():

    for article in iter_articles(ZIM_FILE):
        metadata = extract_metadata(article)
        chunks = chunk_article(metadata)

        print("=" * 80)
        print(metadata.title)
        print()

        print(f"Words: {metadata.word_count}")
        print(f"Links: {metadata.link_count}")
        print(f"Sections: {metadata.section_count}")
        print(f"Chunks: {len(chunks)}")
        print()

        for chunk in chunks:

            print("-" * 60)
            print(f"Section: {chunk.section_title}")
            print(f"Chunk #{chunk.chunk_index}")
            print()

            preview = chunk.text[:250]

            if len(chunk.text) > 250:
                preview += "..."

            print(preview)

        print()


if __name__ == "__main__":
    main()