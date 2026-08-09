"""Generates an answer from a chat-message prompt using a small local LLM.

Backend-agnostic wrapper driven by `config.llm.backend`, taking the message
list built by `rag/prompt_builder.py` and returning generated text -- the
rest of the pipeline (cli.py) never needs to know which backend is active.

- "llama_cpp": loads a local GGUF file directly via `llama-cpp-python`
  (optional dependency, see `pyproject.toml`'s `[llm]` extra). This is what
  actually runs standalone on the Raspberry Pi target: no separate daemon,
  direct control over context size (`config.llm.n_ctx`) and threads.
- "ollama": talks to a running Ollama daemon over its HTTP API
  (POST /api/chat). Nothing to compile, much faster to set up for local dev
  (`ollama pull <model>`) while iterating away from the Pi.

Model loading is lazy and cached per (model_path, n_ctx) -- loading a GGUF is
the expensive part -- mirroring `embeddings/embedder.py`'s `_load_model`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import requests

from wikipod.config import PROJECT_ROOT, LLMConfig


@lru_cache(maxsize=2)
def _load_llama_cpp_model(model_path: str, n_ctx: int):
    from llama_cpp import Llama  # optional dependency; imported lazily on purpose

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"LLM model file not found at {model_path}. Download a GGUF model there "
            "(e.g. from https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), or "
            'switch config.llm.backend to "ollama" for local development.'
        )
    return Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)


def _resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    return p if p.is_absolute() else PROJECT_ROOT / model_path


class Generator:
    """Generates a chat completion from `config.llm`, hiding the backend choice."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.config.backend == "llama_cpp":
            return self._generate_llama_cpp(messages)
        if self.config.backend == "ollama":
            return self._generate_ollama(messages)
        raise ValueError(f"Unknown llm backend: {self.config.backend!r}")

    def _generate_llama_cpp(self, messages: list[dict[str, str]]) -> str:
        if not self.config.model_path:
            raise ValueError("config.llm.model_path is required for the llama_cpp backend")

        model_path = _resolve_model_path(self.config.model_path)
        model = _load_llama_cpp_model(str(model_path), self.config.n_ctx)

        response = model.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _generate_ollama(self, messages: list[dict[str, str]]) -> str:
        if not self.config.ollama_model:
            raise ValueError("config.llm.ollama_model is required for the ollama backend")

        response = requests.post(
            f"{self.config.ollama_host}/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
