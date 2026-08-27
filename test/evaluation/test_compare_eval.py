from io import StringIO

import pytest
from rich.console import Console

from wikipod.evaluation.compare_eval import (
    add_deltas,
    compare_runs,
    extract_summary,
    load_eval_result,
    prepare_run,
    print_comparison,
)


def test_extract_summary_returns_aggregate_metrics():
    result = {
        "k": 5,
        "mean_recall_at_k": 0.8,
        "mean_precision_at_k": 0.16,
        "mean_ndcg_at_k": 0.75,
        "mean_reciprocal_rank": 0.7,
        "per_query": [],
    }

    summary = extract_summary(result)

    assert summary == {
        "k": 5,
        "recall": 0.8,
        "precision": 0.16,
        "ndcg": 0.75,
        "mrr": 0.7,
    }


def test_load_eval_result_reads_json_file(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        """
{
  "k": 5,
  "mean_recall_at_k": 0.8,
  "mean_precision_at_k": 0.16,
  "mean_ndcg_at_k": 0.75,
  "mean_reciprocal_rank": 0.7,
  "per_query": []
}
"""
    )

    result = load_eval_result(result_path)

    assert result["k"] == 5
    assert result["mean_recall_at_k"] == 0.8
    assert result["mean_ndcg_at_k"] == 0.75


def test_prepare_run_uses_parent_directory_as_name(tmp_path):
    run_dir = tmp_path / "baseline"
    run_dir.mkdir()

    result_path = run_dir / "results.json"
    result_path.write_text(
        """
{
  "k": 5,
  "mean_recall_at_k": 0.8,
  "mean_precision_at_k": 0.16,
  "mean_ndcg_at_k": 0.75,
  "mean_reciprocal_rank": 0.7,
  "per_query": []
}
"""
    )

    run = prepare_run(result_path)

    assert run == {
        "name": "baseline",
        "k": 5,
        "recall": 0.8,
        "precision": 0.16,
        "ndcg": 0.75,
        "mrr": 0.7,
    }


def test_compare_runs_prepares_multiple_results(tmp_path):
    baseline_dir = tmp_path / "baseline"
    improved_dir = tmp_path / "improved"
    baseline_dir.mkdir()
    improved_dir.mkdir()

    baseline_path = baseline_dir / "results.json"
    improved_path = improved_dir / "results.json"

    baseline_path.write_text(
        """
{
  "k": 5,
  "mean_recall_at_k": 0.8,
  "mean_precision_at_k": 0.16,
  "mean_ndcg_at_k": 0.75,
  "mean_reciprocal_rank": 0.7,
  "per_query": []
}
"""
    )

    improved_path.write_text(
        """
{
  "k": 5,
  "mean_recall_at_k": 1.0,
  "mean_precision_at_k": 0.2,
  "mean_ndcg_at_k": 0.95,
  "mean_reciprocal_rank": 0.9,
  "per_query": []
}
"""
    )

    runs = compare_runs([baseline_path, improved_path])

    assert len(runs) == 2
    assert runs[0]["name"] == "baseline"
    assert runs[0]["recall"] == 0.8
    assert runs[1]["name"] == "improved"
    assert runs[1]["recall"] == 1.0


def test_add_deltas_compares_runs_against_first_run():
    runs = [
        {
            "name": "baseline",
            "k": 5,
            "recall": 0.9,
            "precision": 0.2,
            "ndcg": 0.8,
            "mrr": 0.85,
        },
        {
            "name": "improved",
            "k": 10,
            "recall": 1.0,
            "precision": 0.1,
            "ndcg": 0.9,
            "mrr": 0.95,
        },
    ]

    result = add_deltas(runs)

    assert result[0]["delta_recall"] == 0.0
    assert result[0]["delta_precision"] == 0.0
    assert result[0]["delta_ndcg"] == 0.0
    assert result[0]["delta_mrr"] == 0.0

    assert result[1]["delta_recall"] == pytest.approx(0.1)
    assert result[1]["delta_precision"] == pytest.approx(-0.1)
    assert result[1]["delta_ndcg"] == pytest.approx(0.1)
    assert result[1]["delta_mrr"] == pytest.approx(0.1)


def test_print_comparison_includes_delta_columns(monkeypatch):
    runs = [
        {
            "name": "baseline",
            "k": 5,
            "recall": 1.0,
            "precision": 0.2,
            "ndcg": 0.96,
            "mrr": 0.95,
            "delta_recall": 0.0,
            "delta_precision": 0.0,
            "delta_ndcg": 0.0,
            "delta_mrr": 0.0,
        },
        {
            "name": "k10",
            "k": 10,
            "recall": 1.0,
            "precision": 0.1,
            "ndcg": 0.96,
            "mrr": 0.95,
            "delta_recall": 0.0,
            "delta_precision": -0.1,
            "delta_ndcg": 0.0,
            "delta_mrr": 0.0,
        },
    ]

    output_buffer = StringIO()

    monkeypatch.setattr(
        "wikipod.evaluation.compare_eval.Console",
        lambda: Console(file=output_buffer, width=200),
    )

    print_comparison(runs)

    output = output_buffer.getvalue()

    assert "baseline" in output
    assert "k10" in output
    assert "-0.100" in output