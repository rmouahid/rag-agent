# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Recorded CLI ingestion and generated-answer transcript with sources and a
  script to reproduce the visual demo (#58).
- Answer-quality results for a real TinyLlama run on all 50 Lumen Rail v2
  questions, with per-question data and explicit model/runtime limitations
  (#81).
- Measured before/after the agentic loop on the same 50 questions: no
  additional retrieval or observed quality improvement on TinyLlama, with
  both raw reports and latency costs published (#82).
- Full-pipeline load measurement runner with p50/p95 latency, peak resident
  memory, per-file provenance and a required post-deduplication corpus size.
  The 200k-chunk production-scale run is still pending (#83).

## [0.2.0] — 2026-11-07

### Added

- Vector indexes: HNSW, product quantization and IVF+PQ next to brute force
  and IVF; small IVF stores fall back to brute force.
- Answering: local generation with llama.cpp (GGUF), token streaming, a
  prompt fitted to the model context window, sources returned with each
  answer, refusal when no retrieved chunk is relevant enough.
- Retrieval pipeline: query rewriting, cross-encoder reranking and an
  agentic loop that searches again when the first context is not enough.
- API: `POST /documents`, `POST /ask`, `/ask/stream` over server-sent
  events, typed schemas, a consistent error envelope and `/health` with
  component readiness. CLI commands `ingest`, `ask` and `eval`.
- Evaluation: a versioned golden dataset (v2, 50 questions), an
  answer-quality harness with rule-based metrics and an optional LLM judge,
  and a retrieval-only mode that needs no LLM.
- Benchmarks: the indexes against brute force and FAISS on SIFT1M, a scale
  study from 10,000 to 100,000 chunks, and retrieval on BEIR SciFact.
- Docker image and Compose file, GitHub Actions CI (tests, ruff, black,
  mypy), pre-commit hooks, contributing guide.

### Changed

- Ingestion goes through one `IngestionPipeline`; near-duplicate chunks are
  found through an inverted shingle index.
- The README leads with the problem, the architecture and measured results;
  the reference manual moves to `docs/guide.md`.

### Fixed

- Request size is capped and uploaded file names are stripped of paths.
- URLs that resolve to non-public addresses are refused.
- The IVF of an IVF+PQ index is saved inside `pq_index.npz`.
- Judge prompts fit the model context window.
- Folders without any supported file are reported.

## [0.1.0] — 2026-09-23

First version: PDF, Markdown and web ingestion, fixed-size and
sentence-aware chunking with deduplication, sentence-transformers
embeddings, a persistent vector store with brute-force and IVF search
(k-means written from scratch, incremental updates) and a brute-force vs
IVF benchmark.

[Unreleased]: https://github.com/rmouahid/rag-agent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rmouahid/rag-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rmouahid/rag-agent/releases/tag/v0.1.0
