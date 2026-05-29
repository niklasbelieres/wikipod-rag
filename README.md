# WikiPod: Document Selection & Retrieval-Augmented Generation

A student project (PIB-PA SoSe 2026) at **HTW Saar** — Hochschule für Technik und Wirtschaft des Saarlandes, supervised by **Prof. Dr.-Ing. Klaus Berberich** (Databases & Information Systems).

## Overview

WikiPod explores how offline access to Wikipedia's knowledge base can be maintained when no direct internet connection is available (e.g. due to technical outages). The project runs on constrained hardware and combines smart document selection with Retrieval-Augmented Generation (RAG).


## Hardware

| Component | Spec |
|-----------|------|
| Device | Raspberry Pi 5 |
| RAM | 16 GB |
| Storage | 1 TB SSD |

## Goals

1. **Local Wikipedia copy** — Set up a local mirror of a subset of the English Wikipedia using [KIWIX](https://kiwix.org)
2. **Document selection** — Develop methods to intelligently select a Wikipedia subset that fits within a given storage budget, considering both article content and metadata (e.g. page views, categories)
3. **Vector indexing** — Index the selected subset as dense embedding-based vectors using [OpenSearch](https://opensearch.org)
4. **Language model selection** — Choose a small language model suitable for the constrained hardware
5. **Evaluation** — Assess the solution with respect to efficiency and effectiveness

## Tech Stack

- [KIWIX](https://kiwix.org) — Offline Wikipedia hosting
- [OpenSearch](https://opensearch.org) — Dense vector indexing & search
- Small LLM (TBD) — On-device inference for answer generation

## Architecture

```
User Query
    │
    ▼
OpenSearch (dense vector retrieval)
    │
    ▼
Retrieved Wikipedia articles
    │
    ▼
Small Language Model (summarization / answer generation)
    │
    ▼
Answer
```

## Team

- Luca Britten
- Jona Mees
- Niklas Bélières