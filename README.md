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

## Evaluation

WikiPod includes a retrieval evaluation workflow based on a fixed YAML dataset of natural-language queries and relevant Wikipedia article titles. The current evaluation reports metrics such as **Recall@k** and **reciprocal rank** so retrieval changes can be compared against a reproducible baseline.

For local evaluation, a small `wikipedia_en_100` ZIM can be used to build a fast test index. Pi-specific runs can use `config/pi-test.yaml` so the default development configuration does not need to be modified for each experiment.

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