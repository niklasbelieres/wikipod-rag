# WikiPod: Document Selection & Retrieval-Augmented Generation

A student project (PIB-PA SoSe 2026) at **HTW Saar** — Hochschule für Technik und Wirtschaft des Saarlandes, supervised by **Prof. Dr.-Ing. Klaus Berberich** (Databases & Information Systems).

## Overview

WikiPod explores how offline access to Wikipedia's knowledge base can be maintained when no direct internet connection is available. The project runs on constrained hardware and combines document selection, dense retrieval, local language-model inference, evaluation, and runtime monitoring.

The current pipeline reads Wikipedia ZIM files, scores and selects articles within a configurable storage budget, chunks the selected content, generates dense embeddings, indexes the chunks in OpenSearch, retrieves relevant chunks for user queries, and optionally uses a small local language model to generate answers from the retrieved context.

## Hardware

| Component | Spec |
|-----------|------|
| Device | Raspberry Pi 5 |
| RAM | 16 GB |
| Storage | 1 TB SSD |

## Goals

1. **Local Wikipedia copy** — Work with offline English Wikipedia snapshots provided as KIWIX/ZIM files
2. **Document selection** — Select a useful Wikipedia subset within a configurable storage budget using signals such as word count, links, incoming links, importance, and page views
3. **Vector indexing** — Chunk selected articles, generate embeddings, and index them in OpenSearch for dense retrieval
4. **Local RAG** — Retrieve relevant Wikipedia chunks and use them as context for a small local language model
5. **Evaluation** — Measure retrieval effectiveness with a fixed query/relevance dataset using metrics such as Recall@k and reciprocal rank
6. **Resource monitoring** — Track RAM, swap, CPU load, temperature, WikiPod RSS, and OpenSearch RSS during indexing runs on the Raspberry Pi

## Tech Stack

- [KIWIX](https://kiwix.org) / ZIM — Offline Wikipedia snapshots
- [OpenSearch](https://opensearch.org) — Dense vector indexing and retrieval
- Sentence Transformers — Embedding generation
- Ollama / local LLM backend — On-device answer generation
- Python — Selection, chunking, indexing, evaluation, and monitoring
- Matplotlib — Plotting monitoring results from CSV logs

## Pipeline

```text
Wikipedia ZIM
    │
    ▼
Metadata extraction + scoring
    │
    ▼
Article selection within storage budget
    │
    ▼
Full-text loading + chunking
    │
    ▼
Embedding generation
    │
    ▼
OpenSearch vector index
    │
    ▼
User Query
    │
    ▼
Dense retrieval of relevant chunks
    │
    ▼
Local LLM with retrieved context
    │
    ▼
Answer
```
## Setup

### Prerequisites

- Python 3.11 or newer (CI tests Python 3.11, 3.12, and 3.13).
- Git and Docker with Docker Compose.
- A local Wikipedia ZIM file and enough storage for the index and models.
- For generated answers: Ollama or the optional `llama-cpp-python` backend.

Run the commands below in a Unix shell from the repository root, with the virtual
environment activated. Initial setup requires internet access to obtain packages,
container images, the Wikipedia corpus, and model files.

### Installation

```bash
git clone https://github.com/niklasbelieres/wikipod-rag.git
cd wikipod-rag
```

Create a virtual environment and install the project with its development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure the corpus

The full Wikipedia ZIM must be provided separately. The repository includes
`test/data/climate-change-mini.zim` for automated tests; it is not the corpus used
for the report evaluation.

Configuration merges `config/default.yaml` with `config/<WIKIPOD_ENV>.yaml`.
The default environment is `dev`. Add or update the following setting in
`config/dev.yaml`, replacing the example with your actual file:

```yaml
paths:
  zim_file: "/absolute/path/to/wikipedia.zim"
```

Relative paths are resolved against the repository root. Use the same environment,
index name, and embedding model for indexing and subsequent queries.

### Start OpenSearch

Start OpenSearch with Docker Compose:

```bash
docker compose up -d
```

Verify that OpenSearch is running:

```bash
curl http://localhost:9200
```

The Compose setup uses OpenSearch 2.17.0, a 2 GB Java heap, and a persistent
Docker volume. It disables authentication and TLS and is intended for local
development. Wait for the HTTP check to succeed before indexing.

### Build an index

Runtime-specific configuration can be selected through `WIKIPOD_ENV`. Once the
configured ZIM file is available and OpenSearch is running, indexing can be
started with:

```bash
WIKIPOD_ENV=dev python -m wikipod.cli index
```

### Rebuilding indices after the chunk-ID change

Chunk IDs now use the article ID, zero-based section position, and chunk position
within that section. This prevents equally named sections from overwriting each
other. The same article with the same section order and chunking settings produces
the same IDs.

Existing indices must be rebuilt once to replace the old IDs and restore chunks
that may have been overwritten. Re-indexing without recreation leaves old documents
in the index. The following command **deletes and rebuilds the configured index**;
select the environment whose index you intend to replace:

```bash
WIKIPOD_ENV=dev python -m wikipod.cli index --recreate-index
```

### Set up answer generation

The `dev`, `report_eval`, and `pi-test` configurations select Ollama. Install it
using the [official Ollama instructions](https://docs.ollama.com/quickstart).
If it is not already running as an application or service, start it in a separate
terminal and leave that terminal open:

```bash
ollama serve
```

Download the model named by `llm.ollama_model` in the configuration:

```bash
ollama pull qwen2.5:1.5b
```

WikiPod connects to `llm.ollama_host`, which defaults to
`http://localhost:11434`. See the [Ollama CLI reference](https://docs.ollama.com/cli)
for service and model commands.

Alternatively, install the optional backend for a local GGUF model:

```bash
pip install -e ".[dev,llm]"
```

Provide a compatible GGUF file separately and set these values in your selected
configuration file:

```yaml
llm:
  backend: llama_cpp
  model_path: "/absolute/path/to/model.gguf"
```

This backend loads the model directly and does not require an Ollama service.

### Querying

After the index has been built, a query can be executed with:

```bash
WIKIPOD_ENV=dev python -m wikipod.cli query "What is the Catholic Church?"
```

To inspect only the retrieved chunks without running the local language model:

```bash
WIKIPOD_ENV=dev python -m wikipod.cli query --chunks-only "What is the Catholic Church?"
```

### Prepare for offline use

Before disconnecting the target machine, install the Python dependencies, obtain
the Docker image and ZIM file, and download the chosen language model. Indexing
and querying also load `embeddings.model_name` (by default
`sentence-transformers/all-MiniLM-L6-v2`); ensure its model files are available
locally by running the pipeline while connected or configuring a local model path.
Use the same embedding model that produced the index.

Verify both retrieval and answer generation on the target machine without internet
access before relying on the offline setup. The report's full-corpus workflow
builds the index on a server and transfers it to the Pi using OpenSearch snapshots;
see report chapters 4 and 6. The Compose file provides a `./snapshots` mount for
this purpose, but copying snapshot files alone does not restore an index.

## Testing

Run the automated test suite and lint checks with:

```bash
pytest
ruff check .
```

The same checks are executed by GitHub Actions for pushes and pull requests to `main`.
CI starts OpenSearch for the integration test. Locally, that test is skipped when
OpenSearch is unavailable. Reader tests using a process pool require an environment
that permits multiprocessing.

## Status

The core WikiPod pipeline is implemented: ZIM analysis and metadata extraction, budget-based document selection, chunking and embedding generation, OpenSearch indexing and dense retrieval, local RAG generation, retrieval evaluation, and Raspberry Pi runtime monitoring.

The project is currently focused on evaluation, deployment measurements, and final documentation.

## Evaluation

Retrieval evaluation deduplicates chunks by article title, retaining the first
chunk per article in the latest search result's ranked order. It starts with
`4 * k` candidates and doubles the requested count until it finds `k` distinct
articles, the backend returns fewer hits than requested, or it reaches the
1,000-candidate limit (`MAX_RETRIEVAL_CANDIDATES` in `run_eval.py`).

If the limit is reached without enough distinct articles, a warning identifies
the query and the number found. Metrics use the available articles without
padding; Precision@k still divides by the requested `k`. Relevant articles beyond
the candidate limit may be missed, so results are bounded-search measurements,
not a guarantee of the top-k distinct articles across the entire index. A shorter
backend response ends the search but does not prove exhaustive coverage of an
approximate nearest-neighbor index. Compared runs should use the same limit.

### Run retrieval evaluation

The dataset contains queries and relevant Wikipedia article titles. Reported
metrics include Recall@k, Precision@k, nDCG@k, and mean reciprocal rank (MRR).

Before evaluating, configure `paths.zim_file` in `config/report_eval.yaml` and
build an index from the intended evaluation corpus. The configured
`wikipedia_en_100_2026-08.zim` file must be supplied separately. `report_eval`
uses its own index, `wikipod-chunks-report-eval`; development continues to use
`wikipod-chunks-dev`. Build the new evaluation index once before running the
commands below. The previously shared development index is not renamed or deleted.
Existing indices containing a different corpus must be recreated deliberately.

```bash
WIKIPOD_ENV=report_eval python -m wikipod.cli index
WIKIPOD_ENV=report_eval python -m wikipod.cli evaluate \
  --dataset test/data/eval_queries.yaml \
  --top-k 5 \
  --output-dir logs/report_eval_k5
```

The output directory contains `results.json` and `results.csv`. Optional flags:

- `--use-query-analyzer`: normalize queries before retrieval.
- `--use-llm-judge`: add model-based relevance judgments; requires the configured
  language-model backend to be ready. Basic retrieval evaluation does not need it.

### Compare local language models

Install `.[dev,llm]`, place compatible `.gguf` files in `models/`, and use a
non-empty evaluation dataset. This command uses `llama_cpp` for each model,
regardless of the backend selected for normal queries:

```bash
WIKIPOD_ENV=report_eval python -m wikipod.cli evaluate-models \
  --dataset test/data/eval_queries.yaml \
  --models-dir models \
  --top-k 5 \
  --output-dir logs/model_comparison
```

Answers and latencies are saved beneath
`<output-dir>/<DD.MM.YY>/<model_stem>/`. Use a separate output directory for each
comparison to avoid overwriting earlier runs on the same day.

## Monitoring

Long-running indexing experiments on the Raspberry Pi can be monitored with `monitor_run.py`, which records system and process metrics to CSV. The monitoring script reads Linux-specific system information such as `/proc/meminfo`, so it is intended to run on the Raspberry Pi or another Linux system.

On the Raspberry Pi / Linux:

```bash
python -m wikipod.scripts.monitor_run \
  --out logs/pi_top100_index.csv \
  --interval 10
```

After the indexing run, stop the monitor with `Ctrl+C`. The resulting CSV can then be analyzed with `plot_metrics.py`, either on the Raspberry Pi or on another machine such as macOS:

```bash
python -m wikipod.scripts.plot_metrics logs/pi_top100_index.csv
```

`plot_metrics.py` prints summary statistics and generates plots for CPU usage, memory usage, swap usage, and CPU temperature.

## Team

- Luca Britten
- Jona Mees
- Niklas Bélières
