"""Compares generation quality/latency of several local SLMs on the same eval dataset.

Unlike `run_eval.py` (retrieval-only metrics), this drives the full RAG
generation step -- but retrieval only depends on the fixed embedding model and
index, not on which SLM answers the question. So retrieval is run exactly
once per query up front, and each candidate model only redoes the (cheap to
compare, expensive to run on a Pi 4) generation step against that same
context.

Models are discovered as `*.gguf` files in a single directory (e.g. mounted
on a Raspberry Pi 4) and run through the existing `llama_cpp` Generator
backend directly -- no Ollama daemon required.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from wikipod.config import get_config
from wikipod.embeddings.embedder import Embedder
from wikipod.evaluation.run_eval import load_eval_dataset
from wikipod.indexing.opensearch_client import build_client
from wikipod.rag.generator import Generator
from wikipod.rag.prompt_builder import build_messages
from wikipod.rag.retriever import Retriever

console = Console()


def prepare_queries(retriever: Retriever, dataset: list[dict], k: int) -> list[dict]:
    """Retrieve context and build the chat prompt once per query.

    Retrieval is independent of which SLM will later answer, so this result
    is computed once and reused across every model.
    """
    prepared = []
    for item in dataset:
        chunks = retriever.retrieve(item["query"], k=k)
        prepared.append(
            {
                "query": item["query"],
                "relevant_titles": item["relevant_titles"],
                "messages": build_messages(item["query"], chunks),
            }
        )
    return prepared


def run_model(generator: Generator, prepared: list[dict]) -> list[dict]:
    """Generate an answer for each prepared query and time it."""
    per_query = []
    for item in prepared:
        start = time.perf_counter()
        answer = generator.generate(item["messages"])
        latency = time.perf_counter() - start
        per_query.append(
            {
                "query": item["query"],
                "relevant_titles": item["relevant_titles"],
                "answer": answer,
                "latency_seconds": latency,
            }
        )
    return per_query


def write_model_results(output_dir: Path, model_name: str, k: int, per_query: list[dict]) -> Path:
    model_dir = output_dir / Path(model_name).stem
    model_dir.mkdir(parents=True, exist_ok=True)

    mean_latency = sum(entry["latency_seconds"] for entry in per_query) / len(per_query)
    results = {
        "model": model_name,
        "k": k,
        "mean_latency_seconds": mean_latency,
        "per_query": per_query,
    }

    with (model_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    with (model_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["query", "relevant_titles", "answer", "latency_seconds"]
        )
        writer.writeheader()
        for entry in per_query:
            row = entry.copy()
            row["relevant_titles"] = "; ".join(entry["relevant_titles"])
            writer.writerow(row)

    return model_dir


@click.command("evaluate-models")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the eval dataset (YAML/JSON list of {query, relevant_titles}).",
)
@click.option(
    "--models-dir",
    "models_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the .gguf SLM files to compare.",
)
@click.option("--top-k", "top_k", default=None, type=int, help="Override config.retrieval.top_k.")
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write one results.json/results.csv subfolder per model.",
)
def main(dataset_path: Path, models_dir: Path, top_k: int | None, output_dir: Path) -> None:
    """Run each .gguf model in MODELS_DIR against DATASET_PATH and record answers/latency."""
    config = get_config()
    k = top_k or config.retrieval.top_k

    embedder = Embedder(config.embeddings.model_name)
    client = build_client(config.opensearch)
    retriever = Retriever(client, config.opensearch.index_name, embedder, top_k=k)

    dataset = load_eval_dataset(dataset_path)
    console.print(f"[bold]{len(dataset)} Queries geladen aus[/bold] {dataset_path}")

    prepared = prepare_queries(retriever, dataset, k)

    model_paths = sorted(models_dir.glob("*.gguf"))
    if not model_paths:
        console.print(f"[red]No .gguf files found in {models_dir}[/red]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_path in model_paths:
        console.print(f"\n[bold]Running {model_path.name}[/bold]")
        llm_config = config.llm.model_copy(
            update={"backend": "llama_cpp", "model_path": str(model_path)}
        )
        generator = Generator(llm_config)

        try:
            per_query = run_model(generator, prepared)
        except Exception as exc:  # noqa: BLE001 - one bad model must not abort the whole run
            console.print(f"[red]Skipping {model_path.name}: {exc}[/red]")
            continue

        model_dir = write_model_results(output_dir, model_path.name, k, per_query)
        mean_latency = sum(entry["latency_seconds"] for entry in per_query) / len(per_query)
        summary_rows.append((model_path.name, mean_latency, len(per_query)))
        console.print(f"[green]Results written to[/green] {model_dir}")

    table = Table(title="SLM Comparison Summary")
    table.add_column("Model")
    table.add_column("Mean Latency (s)")
    table.add_column("Queries")
    for name, mean_latency, count in summary_rows:
        table.add_row(name, f"{mean_latency:.2f}", str(count))
    console.print(table)


if __name__ == "__main__":
    main()
