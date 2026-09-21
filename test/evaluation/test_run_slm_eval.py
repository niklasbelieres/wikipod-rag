from datetime import date
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from rich.console import Console

from wikipod.evaluation.run_slm_eval import main, write_model_results


class _FixedDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 9, 2)


def _patch_pipeline(monkeypatch, models_dir):
    (models_dir / "model-a.gguf").write_bytes(b"")

    monkeypatch.setattr(
        "wikipod.evaluation.run_slm_eval.load_eval_dataset",
        lambda path: [{"query": "q1", "relevant_titles": ["Title"]}],
    )
    monkeypatch.setattr("wikipod.evaluation.run_slm_eval.build_client", lambda config: MagicMock())
    monkeypatch.setattr("wikipod.evaluation.run_slm_eval.Embedder", MagicMock())
    monkeypatch.setattr("wikipod.evaluation.run_slm_eval.Retriever", MagicMock())
    monkeypatch.setattr("wikipod.evaluation.run_slm_eval.Generator", MagicMock())
    monkeypatch.setattr(
        "wikipod.evaluation.run_slm_eval.prepare_queries",
        lambda retriever, dataset, k: [
            {"query": "q1", "relevant_titles": ["Title"], "messages": []}
        ],
    )
    monkeypatch.setattr(
        "wikipod.evaluation.run_slm_eval.run_model",
        lambda generator, prepared: [
            {
                "query": "q1",
                "relevant_titles": ["Title"],
                "answer": "an answer",
                "latency_seconds": 1.0,
            }
        ],
    )
    monkeypatch.setattr("wikipod.evaluation.run_slm_eval.date", _FixedDate)
    monkeypatch.setattr(
        "wikipod.evaluation.run_slm_eval.console", Console(width=200)
    )


def test_main_writes_results_under_dated_subfolder(tmp_path, monkeypatch):
    runner = CliRunner()

    dataset_path = tmp_path / "eval.yaml"
    dataset_path.write_text("- query: q1\n  relevant_titles: [Title]\n")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    output_dir = tmp_path / "eval_results"

    _patch_pipeline(monkeypatch, models_dir)

    result = runner.invoke(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    dated_results = output_dir / "02.09.26" / "model-a" / "results.json"
    undated_results = output_dir / "model-a" / "results.json"

    assert dated_results.exists()
    assert not undated_results.exists()


def test_main_prints_dated_path_in_console_output(tmp_path, monkeypatch):
    runner = CliRunner()

    dataset_path = tmp_path / "eval.yaml"
    dataset_path.write_text("- query: q1\n  relevant_titles: [Title]\n")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    output_dir = tmp_path / "eval_results"

    _patch_pipeline(monkeypatch, models_dir)

    result = runner.invoke(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "02.09.26" in result.output


@pytest.mark.parametrize("content", ["", "[]", "# no queries\n", "null"])
def test_empty_dataset_fails_before_pipeline_setup(tmp_path, monkeypatch, content):
    dataset = tmp_path / "empty.yaml"
    dataset.write_text(content)
    output = tmp_path / "results"
    setup = MagicMock(side_effect=AssertionError("Pipeline must not start"))
    monkeypatch.setattr("wikipod.evaluation.run_slm_eval.get_config", setup)
    result = CliRunner().invoke(main, [
        "--dataset", str(dataset), "--models-dir", str(tmp_path),
        "--output-dir", str(output),
    ])
    assert result.exit_code == 2
    assert "--dataset" in result.output
    assert "at least one query" in result.output
    setup.assert_not_called()
    assert not output.exists()


def test_empty_results_do_not_create_output(tmp_path):
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="without evaluated queries"):
        write_model_results(output, "model.gguf", 5, [])
    assert not output.exists()
