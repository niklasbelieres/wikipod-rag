"""Compare results from multiple retrieval evaluation runs."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def load_eval_result(path: Path) -> dict:
    """Load evaluation results from a JSON file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def extract_summary(result: dict) -> dict:
    """Extract aggregate metrics from one evaluation result."""
    return {
        "k": result["k"],
        "recall": result["mean_recall_at_k"],
        "precision": result["mean_precision_at_k"],
        "ndcg": result["mean_ndcg_at_k"],
        "mrr": result["mean_reciprocal_rank"],
    }


def prepare_run(path: Path) -> dict:
    """Load one evaluation run and prepare it for comparison."""
    result = load_eval_result(path)
    summary = extract_summary(result)

    return {
        "name": path.parent.name,
        **summary,
    }


def compare_runs(paths: list[Path]) -> list[dict]:
    """Load and prepare multiple evaluation runs."""
    return [prepare_run(path) for path in paths]


def add_deltas(runs: list[dict]) -> list[dict]:
    """Add metric differences relative to the first evaluation run."""
    if not runs:
        return []

    baseline = runs[0]
    metrics = ("recall", "precision", "ndcg", "mrr")
    runs_with_deltas = []

    for run in runs:
        run_with_deltas = run.copy()

        for metric in metrics:
            run_with_deltas[f"delta_{metric}"] = (
                run[metric] - baseline[metric]
            )

        runs_with_deltas.append(run_with_deltas)

    return runs_with_deltas


def print_comparison(runs: list[dict]) -> None:
    """Print evaluation runs as a comparison table."""
    table = Table(title="Evaluation Run Comparison")

    table.add_column("Run")
    table.add_column("k")
    table.add_column("Recall@k")
    table.add_column("Δ Recall")
    table.add_column("Precision@k")
    table.add_column("Δ Precision")
    table.add_column("nDCG@k")
    table.add_column("Δ nDCG")
    table.add_column("MRR")
    table.add_column("Δ MRR")

    for run in runs:
        table.add_row(
            run["name"],
            str(run["k"]),
            f"{run['recall']:.3f}",
            f"{run['delta_recall']:+.3f}",
            f"{run['precision']:.3f}",
            f"{run['delta_precision']:+.3f}",
            f"{run['ndcg']:.3f}",
            f"{run['delta_ndcg']:+.3f}",
            f"{run['mrr']:.3f}",
            f"{run['delta_mrr']:+.3f}",
        )

    Console().print(table)


def main() -> None:
    """Compare evaluation result files from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare multiple retrieval evaluation runs."
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="Paths to evaluation results.json files.",
    )

    args = parser.parse_args()

    runs = compare_runs(args.results)
    runs = add_deltas(runs)
    print_comparison(runs)


if __name__ == "__main__":
    main()