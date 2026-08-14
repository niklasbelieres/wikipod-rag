"""Builds the LLM prompt from a query and the retrieved chunks.

Pure text-assembly step between `rag/retriever.py` (query -> ranked `Chunk`s)
and `rag/generator.py` (prompt -> generated answer). No model loading, no
network I/O here -- that's what keeps this file trivially unit-testable
without a running backend.

Output format: a list of chat messages (`[{"role": ..., "content": ...}]`),
not a single flat string. Both backends in `config.llm.backend` speak this
format (llama-cpp-python's `create_chat_completion(messages=...)` and
Ollama's `POST /api/chat`), so building messages here keeps this module
backend-agnostic.
"""

from __future__ import annotations

from wikipod.chunking.models import Chunk

SYSTEM_PROMPT = (
    "You are an assistant answering questions using only the Wikipedia excerpts "
    "given below as context. Answer using only information contained in the "
    "excerpts. If the excerpts do not contain enough information to answer, say "
    "so explicitly instead of guessing. Keep answers concise, and refer to "
    "excerpts by their [n] number when useful."
)

# Word-count proxy for the model's context budget. sentence-transformers/GGUF
# tokenizers run roughly ~1.3 tokens/word for English, so this leaves headroom
# under config.llm.n_ctx (default 4096) once the system prompt and
# config.llm.max_tokens (default 400) worth of output are accounted for.
DEFAULT_MAX_CONTEXT_WORDS = 800

def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a numbered, source-attributed context block.

    Example:
        [1] (Climate change - Causes) Human activity is the main driver...
        [2] (Climate change - Lead) Present-day climate change includes...
    """
    lines = [
        f"[{i}] ({chunk.article_title} - {chunk.section_title}) {chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(lines)


def _fit_to_budget(chunks: list[Chunk], max_context_words: int) -> list[Chunk]:
    """Trim `chunks` (already ranked best-first by the retriever) to fit a word budget.

    Same "greedy, take while it still fits" shape as `select_within_budget` in
    `selection/selector.py`: skip anything that would blow the budget rather
    than stopping at the first one that doesn't fit, so one oversized chunk
    doesn't crowd out smaller ones ranked right after it.
    """
    selected: list[Chunk] = []
    used_words = 0

    for chunk in chunks:
        if used_words + chunk.word_count > max_context_words:
            continue
        selected.append(chunk)
        used_words += chunk.word_count

    return selected


def build_messages(
    query: str,
    chunks: list[Chunk],
    max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS,
) -> list[dict[str, str]]:
    """Assemble the full chat-message list to send to the generator.

    If `chunks` is empty (no retrieval hits), the prompt still goes out --
    the system prompt already instructs the model to say so rather than
    guess, so there's no need for the caller to special-case an empty
    retrieval result before calling this.
    """
    fitted = _fit_to_budget(chunks, max_context_words)
    context = (
        format_context(fitted)
        if fitted
        else "No relevant excerpts were found in the indexed Wikipedia subset."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    