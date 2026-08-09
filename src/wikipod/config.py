"""Configuration loading.

Merges ``config/default.yaml`` with an environment-specific override file
(``config/dev.yaml`` or ``config/prod.yaml``), selected via the
``WIKIPOD_ENV`` environment variable (defaults to ``dev``).

Usage:
    >>> from wikipod.config import get_config
    >>> cfg = get_config()
    >>> cfg.selection.storage_budget_mb
    50
"""

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class PathsConfig(BaseModel):
    zim_file: str
    pageviews_file: str | None = None
    data_dir: str = "data"


class SelectionWeights(BaseModel):
    word_count: float = 0.2
    link_count: float = 0.2
    importance: float = 0.2
    incoming_links: float = 0.4
    pageviews: float = 0.0


class SelectionConfig(BaseModel):
    storage_budget_mb: float
    weights: SelectionWeights = SelectionWeights()


class ChunkingConfig(BaseModel):
    max_words: int = 250
    overlap: int = 40


class EmbeddingsConfig(BaseModel):
    model_name: str
    batch_size: int = 32
    dimension: int = 384


class OpenSearchConfig(BaseModel):
    host: str = "localhost"
    port: int = 9200
    use_ssl: bool = False
    verify_certs: bool = False
    index_name: str = "wikipod-chunks"


class LLMConfig(BaseModel):
    backend: str = "llama_cpp"
    model_path: str | None = None
    ollama_model: str | None = None
    ollama_host: str = "http://localhost:11434"
    max_tokens: int = 400
    temperature: float = 0.2
    n_ctx: int = 4096  # llama_cpp only; must cover system prompt + context + max_tokens


class RetrievalConfig(BaseModel):
    top_k: int = 5


class WikipodConfig(BaseModel):
    paths: PathsConfig
    selection: SelectionConfig
    chunking: ChunkingConfig = ChunkingConfig()
    embeddings: EmbeddingsConfig
    opensearch: OpenSearchConfig = OpenSearchConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config path relative to the project root, unless it's already absolute."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@cache
def get_config(env: str | None = None) -> WikipodConfig:
    """Load and cache the merged configuration for the given environment.

    Args:
        env: "dev" or "prod". Falls back to the ``WIKIPOD_ENV`` env var, then "dev".
    """
    env = env or os.environ.get("WIKIPOD_ENV", "dev")

    base = _load_yaml(CONFIG_DIR / "default.yaml")
    override = _load_yaml(CONFIG_DIR / f"{env}.yaml")
    merged = _deep_merge(base, override)

    return WikipodConfig(**merged)
