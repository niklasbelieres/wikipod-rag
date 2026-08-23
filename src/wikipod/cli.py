"""Command-line entry point for the WikiPod pipeline.

Wraps the analysis -> selection -> chunking -> embedding -> indexing pipeline
behind `wikipod index`, and retrieval + answer generation against an
already-built index behind `wikipod query`. Both commands are driven by the
merged config from `wikipod.config` (see `config/default.yaml`, overridden
per `WIKIPOD_ENV`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from wikipod.analysis.reader import (
    read_articles_metadata_cached,
    read_articles_metadata_for_ids,
    read_articles_metadata_parallel,
)
from wikipod.analysis.statistics import link_frequency_map
from wikipod.chunking.chunker import chunk_article
from wikipod.config import get_config
from wikipod.embeddings.embedder import Embedder
from wikipod.indexing.opensearch_client import build_client, create_index, index_chunks
from wikipod.rag.retriever import Retriever
from wikipod.rag.prompt_builder import build_messages
from wikipod.rag.generator import Generator
from wikipod.selection.pageviews import download_pageviews_day, load_pageviews, save_pageviews
from wikipod.selection.selector import select_within_budget

console = Console()


@click.group()
def cli() -> None:
    """WikiPod: offline Wikipedia selection, indexing and retrieval."""


@cli.command("fetch-pageviews")
@click.option("--date", "date_str", required=True, help="Date to fetch, as YYYY-MM-DD.")
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output path (default: config.paths.pageviews_file, or ./pageviews.txt).",
)
def fetch_pageviews(date_str: str, out_path: str | None) -> None:
    """Download and aggregate a full day of Wikimedia pageviews for selection scoring."""
    config = get_config()
    day = datetime.strptime(date_str, "%Y-%m-%d").date()
    target = Path(out_path) if out_path else config.resolve_path(
        config.paths.pageviews_file or "pageviews.txt"
    )

    console.print(f"[bold]Fetching pageviews for {day}[/bold] (24 hourly dumps)")
    with Progress(console=console) as progress:
        task = progress.add_task("Downloading hours", total=24)

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        views = download_pageviews_day(day, on_progress=on_progress)

    save_pageviews(views, target)
    console.print(f"[green]Wrote {len(views)} article view-counts to {target}[/green]")


@cli.command()
@click.option(
    "--recreate-index", is_flag=True, help="Drop and recreate the OpenSearch index first."
)
@click.option("--workers", default=None, type=int, help="Parallel workers for reading (default: all cores).")
@click.option(
    "--no-cache", is_flag=True, help="Re-parse the ZIM even if a metadata cache exists."
)
def index(recreate_index: bool, workers: int, no_cache: bool) -> None:
    """Select articles within budget, chunk, embed and index them into OpenSearch."""
    config = get_config()
    zim_path = config.resolve_path(config.paths.zim_file)

    console.print(f"[bold]Reading articles from[/bold] {zim_path}")
    with Progress(console=console) as progress:
        task = progress.add_task("Reading articles", total=None)

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        if no_cache:
            articles = read_articles_metadata_parallel(
                zim_path, workers, on_progress=on_progress, include_sections=False
            )
        else:
            cache_path = (
                config.resolve_path(config.paths.data_dir) / f"{zim_path.stem}_metadata_cache.pkl"
            )
            articles = read_articles_metadata_cached(
                zim_path, cache_path, workers, on_progress=on_progress, include_sections=False
            )
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

    console.print("[bold]Fetching full text for the selected subset[/bold]")
    selected_ids = [article.article_id for article in result.selected]
    selected_with_text = read_articles_metadata_for_ids(zim_path, selected_ids, workers)

    chunks = [
        chunk
        for article in selected_with_text
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
    with Progress(console=console) as progress:
        task = progress.add_task("Embedding & indexing", total=len(chunks))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.embed_chunks(batch)
            indexed += index_chunks(client, config.opensearch.index_name, batch, vectors)
            progress.update(task, completed=min(start + batch_size, len(chunks)))

    console.print(f"[green]Indexed {indexed} chunks into '{config.opensearch.index_name}'[/green]")


@cli.command()
@click.argument("text")
@click.option("--top-k", default=None, type=int, help="Override config.retrieval.top_k.")
@click.option(
    "--chunks-only", is_flag=True, help="Only show retrieved chunks, skip LLM generation."
)
def query(text: str, top_k: int | None, chunks_only: bool) -> None:
    """Retrieve the top-k chunks most relevant to TEXT and generate an answer."""
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

    if chunks_only:
        return

    messages = build_messages(text, chunks)
    answer = Generator(config.llm).generate(messages)
    console.print(f"\n[bold]Answer:[/bold] {answer}")


if __name__ == "__main__":
    cli()
