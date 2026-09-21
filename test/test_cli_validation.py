import pytest
from click.testing import CliRunner

from wikipod.cli import query
from wikipod.evaluation.run_eval import main as evaluate
from wikipod.evaluation.run_slm_eval import main as evaluate_models


@pytest.mark.parametrize("command", [query, evaluate, evaluate_models])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_invalid_top_k_before_loading_config(command, value, monkeypatch):
    def unexpected_config_load():
        pytest.fail("Invalid CLI input must be rejected before loading configuration")

    monkeypatch.setattr(f"{command.callback.__module__}.get_config", unexpected_config_load)
    result = CliRunner().invoke(command, ["--top-k", value])
    assert result.exit_code == 2
    assert "--top-k" in result.output
    assert "Invalid value" in result.output
    assert "not in the range" in result.output


@pytest.mark.parametrize("command", [query, evaluate, evaluate_models])
@pytest.mark.parametrize("value", [None, "1"])
def test_cli_accepts_smallest_top_k_and_default(command, value, tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("[]")
    args = ["question"] if command is query else ["--dataset", str(dataset)]
    if command is evaluate_models:
        args += ["--models-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    if value is not None:
        args += ["--top-k", value]
    received = {}

    def capture_callback(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(command, "callback", capture_callback)
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    assert received["top_k"] == (None if value is None else 1)
