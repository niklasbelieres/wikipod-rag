import json

import pytest

from wikipod.evaluation.report_models import (
    build_summary_rows,
    find_latest_run_dir,
    group_by_query,
    load_model_result,
    load_run_results,
    parse_run_date,
    render_query_card,
    render_report,
    write_report,
)


# -- parse_run_date -----------------------------------------------------------
def test_parse_run_date_parses_valid_folder_name():
    parsed = parse_run_date("02.09.26")
    assert parsed.day == 2
    assert parsed.month == 9
    assert parsed.year == 2026


def test_parse_run_date_returns_none_for_invalid_name():
    assert parse_run_date("not-a-date") is None
    assert parse_run_date("Phi-3.5-mini-instruct-Q4_K_M") is None


# -- find_latest_run_dir -------------------------------------------------------
def test_find_latest_run_dir_picks_latest_by_parsed_date(tmp_path):
    (tmp_path / "31.08.26").mkdir()
    (tmp_path / "05.09.26").mkdir()

    latest = find_latest_run_dir(tmp_path)

    assert latest.name == "05.09.26"


def test_find_latest_run_dir_falls_back_to_mtime_for_unparseable_names(tmp_path):
    dated = tmp_path / "01.01.26"
    dated.mkdir()
    other = tmp_path / "manual-run"
    other.mkdir()

    latest = find_latest_run_dir(tmp_path)

    assert latest.name == "01.01.26"


def test_find_latest_run_dir_raises_when_no_subfolders(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_run_dir(tmp_path)


# -- load_model_result / load_run_results --------------------------------------
def _write_results(model_dir, model_name, mean_latency, per_query):
    model_dir.mkdir(parents=True)
    (model_dir / "results.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "k": 5,
                "mean_latency_seconds": mean_latency,
                "per_query": per_query,
            }
        )
    )


def test_load_model_result_reads_json_file(tmp_path):
    model_dir = tmp_path / "model-a"
    _write_results(model_dir, "model-a.gguf", 1.5, [])

    result = load_model_result(model_dir / "results.json")

    assert result["model"] == "model-a.gguf"
    assert result["mean_latency_seconds"] == 1.5


def test_load_run_results_reads_all_model_jsons_sorted_by_latency(tmp_path):
    _write_results(tmp_path / "slow-model", "slow-model.gguf", 10.0, [])
    _write_results(tmp_path / "fast-model", "fast-model.gguf", 2.0, [])

    results = load_run_results(tmp_path)

    assert [r["model"] for r in results] == ["fast-model.gguf", "slow-model.gguf"]


def test_load_run_results_skips_dirs_without_results_json(tmp_path):
    _write_results(tmp_path / "model-a", "model-a.gguf", 1.0, [])
    (tmp_path / "empty-dir").mkdir()

    results = load_run_results(tmp_path)

    assert len(results) == 1
    assert results[0]["model"] == "model-a.gguf"


# -- build_summary_rows ---------------------------------------------------------
def test_build_summary_rows_extracts_expected_fields():
    results = [
        {
            "model": "model-a.gguf",
            "mean_latency_seconds": 3.2,
            "per_query": [{"query": "q1"}, {"query": "q2"}],
        }
    ]

    rows = build_summary_rows(results)

    assert rows == [
        {"model": "model-a.gguf", "mean_latency_seconds": 3.2, "query_count": 2}
    ]


# -- group_by_query ---------------------------------------------------------------
def test_group_by_query_pivots_per_model_answers_under_shared_query():
    results = [
        {
            "model": "model-a.gguf",
            "per_query": [
                {
                    "query": "who was Einstein",
                    "relevant_titles": ["Albert Einstein"],
                    "answer": "answer-a",
                    "latency_seconds": 1.0,
                }
            ],
        },
        {
            "model": "model-b.gguf",
            "per_query": [
                {
                    "query": "who was Einstein",
                    "relevant_titles": ["Albert Einstein"],
                    "answer": "answer-b",
                    "latency_seconds": 2.0,
                }
            ],
        },
    ]

    grouped = group_by_query(results)

    assert len(grouped) == 1
    entry = grouped[0]
    assert entry["query"] == "who was Einstein"
    assert entry["relevant_titles"] == ["Albert Einstein"]
    assert entry["by_model"]["model-a.gguf"]["answer"] == "answer-a"
    assert entry["by_model"]["model-b.gguf"]["answer"] == "answer-b"


# -- render_query_card / render_report -------------------------------------------
def test_render_query_card_html_escapes_answer_text():
    entry = {
        "query": "q1",
        "relevant_titles": ["Title"],
        "by_model": {
            "model-a.gguf": {"answer": "<script>alert(1)</script> & more", "latency_seconds": 1.0}
        },
    }

    rendered = render_query_card(entry, ["model-a.gguf"])

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&amp;" in rendered


def test_render_report_includes_summary_table_and_all_model_names():
    results = [
        {
            "model": "model-a.gguf",
            "mean_latency_seconds": 1.0,
            "per_query": [
                {
                    "query": "q1",
                    "relevant_titles": ["Title"],
                    "answer": "answer-a",
                    "latency_seconds": 1.0,
                }
            ],
        },
        {
            "model": "model-b.gguf",
            "mean_latency_seconds": 2.0,
            "per_query": [
                {
                    "query": "q1",
                    "relevant_titles": ["Title"],
                    "answer": "answer-b",
                    "latency_seconds": 2.0,
                }
            ],
        },
    ]

    rendered = render_report(results, run_label="02.09.26")

    assert "model-a.gguf" in rendered
    assert "model-b.gguf" in rendered
    assert "02.09.26" in rendered
    assert "answer-a" in rendered
    assert "answer-b" in rendered


def test_render_report_is_self_contained():
    rendered = render_report([], run_label="02.09.26")

    assert "<link" not in rendered
    assert "<script src" not in rendered
    assert "http://" not in rendered
    assert "https://" not in rendered


# -- write_report -----------------------------------------------------------------
def test_write_report_defaults_to_run_dir_report_html(tmp_path):
    run_dir = tmp_path / "02.09.26"
    run_dir.mkdir()

    target = write_report("<html></html>", run_dir, None)

    assert target == run_dir / "report.html"
    assert target.read_text() == "<html></html>"


def test_write_report_honors_explicit_output_path(tmp_path):
    run_dir = tmp_path / "02.09.26"
    run_dir.mkdir()
    output = tmp_path / "custom.html"

    target = write_report("<html></html>", run_dir, output)

    assert target == output
    assert target.read_text() == "<html></html>"
