import pytest
from pydantic import ValidationError

from wikipod.config import ChunkingConfig, EmbeddingsConfig, RetrievalConfig, get_config


@pytest.mark.parametrize("value", [0, -1])
def test_positive_configuration_values_reject_zero_and_negative(value):
    for model, kwargs, field in [
        (ChunkingConfig, {"max_words": value, "overlap": 0}, "max_words"),
        (EmbeddingsConfig, {"model_name": "test", "batch_size": value}, "batch_size"),
        (RetrievalConfig, {"top_k": value}, "top_k"),
    ]:
        with pytest.raises(ValidationError, match=field):
            model(**kwargs)


@pytest.mark.parametrize("overlap", [-1, 3, 4])
def test_chunking_config_rejects_invalid_overlap(overlap):
    with pytest.raises(ValidationError, match="overlap"):
        ChunkingConfig(max_words=3, overlap=overlap)


@pytest.mark.parametrize("max_words,overlap", [(1, 0), (3, 0), (3, 2)])
def test_configuration_accepts_valid_boundaries(max_words, overlap):
    assert ChunkingConfig(max_words=max_words, overlap=overlap).overlap == overlap
    assert EmbeddingsConfig(model_name="test", batch_size=1).batch_size == 1
    assert RetrievalConfig(top_k=1).top_k == 1


def test_merged_configuration_validates_default_overlap(tmp_path, monkeypatch):
    (tmp_path / "default.yaml").write_text(
        "paths:\n  zim_file: test.zim\nselection:\n  storage_budget_mb: 1\n"
        "embeddings:\n  model_name: test\nchunking:\n  max_words: 250\n  overlap: 40\n"
    )
    (tmp_path / "test.yaml").write_text("chunking:\n  max_words: 20\n")
    monkeypatch.setattr("wikipod.config.CONFIG_DIR", tmp_path)
    get_config.cache_clear()
    try:
        with pytest.raises(ValidationError, match="overlap must be smaller than max_words"):
            get_config("test")
    finally:
        get_config.cache_clear()
