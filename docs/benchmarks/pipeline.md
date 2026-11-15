# Full-pipeline load measurement

`benchmarks/pipeline_load.py` measures the real loader, chunking,
deduplication, embeddings and index, followed by retrieval, the agentic loop
and generation. It uses an in-memory index and leaves the application's
persisted index untouched. It requires the configured GGUF and embedding
models; it has no mock or token-only mode.

Supply a representative Markdown/PDF corpus and a JSONL file with objects
such as `{"question": "Which escalation policy applies to incident P1?"}`.
Use questions supported by that corpus, including several multi-document
questions. Do not replicate a few documents: deduplication removes copies
and a synthetic corpus does not establish production behaviour.

From the repository root:

```bash
poetry run python -m benchmarks.pipeline_load \
  --corpus /path/to/corpus \
  --questions /path/to/questions.jsonl \
  --min-chunks 200000 --repeats 3 \
  --output eval/runs/pipeline-200k.json
```

The default enables the agentic loop; `--no-agentic` is an explicit ablation.
The report records the actual configuration, machine, file hashes, distinct
chunks after deduplication, initialization and ingestion time, p50/p95 query
latency, per-request search counts/statuses and process peak resident memory.
Latencies include retrieval, extra searches and complete generation. The
first request is included; requests run sequentially, so this measures one
active user rather than concurrent service throughput.

Reports are checkpointed after each source and each request, with an explicit
`running`, `failed` or `completed` status. Fewer than the required number of
distinct chunks fails the run. An all-refusal run also fails because it did
not exercise generation. Reusing an output path is rejected to preserve
earlier measurements.

For an engineering smoke test, set `--min-chunks 1` and use a small fixture.
Such a run verifies the tool only; it is not evidence for issue #83. A release
needs a completed report at the requested scale, interpretation of its
bottlenecks, and the exact corpus/question provenance. No 200k full-pipeline
result is claimed until that run has completed.
