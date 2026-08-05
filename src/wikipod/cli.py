"""Command-line entry point for the WikiPod pipeline.

Wraps the analysis -> selection -> chunking -> embedding -> indexing pipeline
behind `wikipod index`, and ad-hoc retrieval against an already-built index
behind `wikipod query`. Both commands are driven by the merged config from
`wikipod.config` (see `config/default.yaml`, overridden per `WIKIPOD_ENV`).

Generation (`rag/generator.py`) isn't wired in yet, so `query` only shows
the retrieved chunks, not a synthesized answer.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from wikipod.analysis.metadata import extract_metadata
from wikipod.analysis.reader import iter_articles
from wikipod.analysis.statistics import link_frequency_map
from wikipod.chunking.chunker import chunk_article
from wikipod.config import get_config
from wikipod.embeddings.embedder import Embedder
from wikipod.indexing.opensearch_client import build_client, create_index, index_chunks
from wikipod.rag.retriever import Retriever
from wikipod.selection.pageviews import load_pageviews
from wikipod.selection.selector import select_within_budget

console = Console()


@click.group()
def cli() -> None:
    """WikiPod: offline Wikipedia selection, indexing and retrieval."""


@cli.command()
@click.option(
    "--recreate-index", is_flag=True, help="Drop and recreate the OpenSearch index first."
)
def index(recreate_index: bool) -> None:
    """Select articles within budget, chunk, embed and index them into OpenSearch."""
    config = get_config()
    zim_path = config.resolve_path(config.paths.zim_file)

    console.print(f"[bold]Reading articles from[/bold] {zim_path}")
    articles = [extract_metadata(a) for a in iter_articles(zim_path)]
    console.print(f"Read {len(articles)} articles")

    link_frequencies = link_frequency_map(articles)

    pageviews: dict[str, int] = {}
    if config.paths.pageviews_file:
        pageviews = load_pageviews(config.resolve_path(config.paths.pageviews_file))

    result = select_within_budget(
        articles,
        link_frequencies,
        config.selection.weights,
        config.selection.storage_budget_mb,
        pageviews=pageviews,
    )
    console.print(
        f"Selected {len(result.selected)}/{result.total_candidates} articles "
        f"({result.used_mb:.1f}/{result.budget_mb:.0f} MB, "
        f"{result.utilization:.0%} utilization)"
    )

    chunks = [
        chunk
        for article in result.selected
        for chunk in chunk_article(
            article, max_words=config.chunking.max_words, overlap=config.chunking.overlap
        )
    ]
    console.print(f"Produced {len(chunks)} chunks")

    embedder = Embedder(config.embeddings.model_name, batch_size=config.embeddings.batch_size)
    client = build_client(config.opensearch)
    create_index(client, config.opensearch.index_name, embedder.dimension, recreate=recreate_index)

    indexed = 0
    batch_size = config.embeddings.batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_chunks(batch)
        indexed += index_chunks(client, config.opensearch.index_name, batch, vectors)

    console.print(f"[green]Indexed {indexed} chunks into '{config.opensearch.index_name}'[/green]")


@cli.command()
@click.argument("text")
@click.option("--top-k", default=None, type=int, help="Override config.retrieval.top_k.")
def query(text: str, top_k: int | None) -> None:
    """Retrieve the top-k chunks most relevant to TEXT (no LLM generation yet)."""
    config = get_config()
    embedder = Embedder(config.embeddings.model_name)
    client = build_client(config.opensearch)
    retriever = Retriever(
        client, config.opensearch.index_name, embedder, top_k=config.retrieval.top_k
    )

    chunks = retriever.retrieve(text, k=top_k)

    table = Table(title=f"Top {len(chunks)} results for: {text}")
    table.add_column("Article")
    table.add_column("Section")
    table.add_column("Preview")
    for chunk in chunks:
        preview = chunk.text[:120] + ("..." if len(chunk.text) > 120 else "")
        table.add_row(chunk.article_title, chunk.section_title, preview)
    console.print(table)


if __name__ == "__main__":
    cli()
