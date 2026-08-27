import csv
import json
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from wikipod.config import get_config
from wikipod.embeddings.embedder import Embedder
from wikipod.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from wikipod.indexing.opensearch_client import build_client
from wikipod.rag.retriever import Retriever

console = Console()

def load_eval_dataset(path) -> list[dict]:
    """Lädt die Query/relevant_titles-Paare aus der JSON/YAML-Datei."""    
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or []


def run_single_query(retriever: Retriever, query: str, k: int) -> list[str]:
    """Returns the top-k unique article titles for a query."""
    chunks = retriever.retrieve(query, k=k * 4)

    unique_titles = list(
        dict.fromkeys(chunk.article_title for chunk in chunks)
    )

    return unique_titles[:k]
    


def run_eval(retriever: Retriever, dataset: list[dict], k: int) -> dict:
    """Run retrieval evaluation for all queries in the dataset.
    Calculates Recall@k, Precision@k, nDCG@k, and reciprocal rank per query,
    plus mean aggregate metrics across the full dataset.
    """
    if not dataset:
        return {"k": k, "mean_recall_at_k": 0.0, "mean_reciprocal_rank": 0.0, "per_query": []}

    queries = [elem["query"] for elem in dataset]
    # recall_at_k/reciprocal_rank erwarten relevant_ids als set (Mitgliedschaftstest),
    # nicht als list wie im Dataset -- muss hier explizit konvertiert werden.
    relevant_title_sets = [set(elem["relevant_titles"]) for elem in dataset]

    retrieved_titles = [run_single_query(retriever, query, k) for query in queries]

    # Pro Query einzeln aufrufen, nicht mit allen Queries auf einmal -- beide
    # Funktionen sind für genau eine Query definiert (siehe Signaturen in metrics.py).
    # strict=True ueberall: alle vier Listen sind per Konstruktion (Listcomps
    # ueber dasselbe `dataset`) gleich lang -- ein Laengen-Mismatch waere ein
    # echter Bug, den strict=True sofort als ValueError sichtbar macht statt
    # ihn still zu verschlucken.
    recall_at_k_values = [
        recall_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(retrieved_titles, relevant_title_sets, strict=True)
    ]
    precision_at_k_values = [
        precision_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(
            retrieved_titles,
            relevant_title_sets,
            strict=True,
        )
    ]

    ndcg_at_k_values = [
        ndcg_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(
            retrieved_titles,
            relevant_title_sets,
            strict=True,
        )
    ]

    reciprocal_rank_values = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_titles, relevant_title_sets, strict=True)
    ]

    per_query = [
        {
            "query": query,
            "relevant_titles": sorted(relevant),
            "retrieved_titles": retrieved,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "ndcg_at_k": ndcg,
            "reciprocal_rank": rank,
        }
        for query, relevant, retrieved, recall, precision, ndcg, rank in zip(
            queries,
            relevant_title_sets,
            retrieved_titles,
            recall_at_k_values,
            precision_at_k_values,
            ndcg_at_k_values,
            reciprocal_rank_values,
            strict=True,
        )
    ]

    return {
        "k": k,
        "mean_recall_at_k": sum(recall_at_k_values) / len(recall_at_k_values),
        "mean_precision_at_k": sum(precision_at_k_values) / len(precision_at_k_values),
        "mean_ndcg_at_k": sum(ndcg_at_k_values) / len(ndcg_at_k_values),
        "mean_reciprocal_rank": mean_reciprocal_rank(
            retrieved_titles,
            relevant_title_sets,
        ),
        "per_query": per_query,
    }


@click.command("evaluate")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the eval dataset (YAML/JSON list of {query, relevant_titles}).",
)
@click.option("--top-k", "top_k", default=None, type=int, help="Override config.retrieval.top_k.")

@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for writing evaluation results as JSON.",
)

@click.option(
    "--csv-output",
    "csv_output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for writing per-query evaluation results as CSV.",
)

def main(
    dataset_path: Path,
    top_k: int | None,
    output_path: Path | None,
    csv_output_path: Path | None,
) -> None:
    """Run retrieval evaluation against DATASET_PATH and print all metrics."""
    config = get_config()
    k = top_k or config.retrieval.top_k

    embedder = Embedder(config.embeddings.model_name)
    client = build_client(config.opensearch)
    retriever = Retriever(client, config.opensearch.index_name, embedder, top_k=k)

    dataset = load_eval_dataset(dataset_path)
    console.print(f"[bold]{len(dataset)} Queries geladen aus[/bold] {dataset_path}")

    results = run_eval(retriever, dataset, k)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

        console.print(f"[green]Evaluation results written to[/green] {output_path}")
    if csv_output_path is not None:
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "query",
            "relevant_titles",
            "retrieved_titles",
            "recall_at_k",
            "precision_at_k",
            "ndcg_at_k",
            "reciprocal_rank",
        ]

        with csv_output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for entry in results["per_query"]:
                row = entry.copy()
                row["relevant_titles"] = "; ".join(entry["relevant_titles"])
                row["retrieved_titles"] = "; ".join(entry["retrieved_titles"])
                writer.writerow(row)

        console.print(
            f"[green]Per-query evaluation results written to[/green] {csv_output_path}"
        )

    table = Table(title=f"Evaluation (k={results['k']})")
    table.add_column("Query")
    table.add_column("Recall@k")
    table.add_column("Precision@k")
    table.add_column("nDCG@k")
    table.add_column("Reciprocal Rank")
    for entry in results["per_query"]:
        table.add_row(
            entry["query"],
            f"{entry['recall_at_k']:.2f}",
            f"{entry['precision_at_k']:.2f}",
            f"{entry['ndcg_at_k']:.2f}",
            f"{entry['reciprocal_rank']:.2f}",
        )
    console.print(table)

    console.print(
        f"\n[bold green]Mean Recall@k:[/bold green] {results['mean_recall_at_k']:.3f}   "
        f"[bold green]Mean Precision@k:[/bold green] {results['mean_precision_at_k']:.3f}   "
        f"[bold green]Mean nDCG@k:[/bold green] {results['mean_ndcg_at_k']:.3f}   "
        f"[bold green]MRR:[/bold green] {results['mean_reciprocal_rank']:.3f}"
    )


if __name__ == "__main__":
    main()