# RAG Agent

A minimal but precise Retrieval-Augmented Generation (RAG) agent built from scratch in Python — no LangChain, no managed vector database, no external ANN library.

## Features

- Multi-source ingestion (PDF, Markdown, Web)
- Custom chunking & embedding pipeline, built without a framework
- Vector search built from scratch with numpy: exact brute-force cosine search, and an IVF index (own k-means, configurable `nprobe`, incremental updates)
- Answer generation via a locally-hosted LLM — no external inference API

## Stack

- Python 3.11+
- Local LLM runtime
- sentence-transformers (embeddings)
- numpy (vector search)

## Package layout

```
rag_agent/
├── ingestion/    # Source loaders (PDF, Markdown, web)
├── chunking/     # Text chunking
├── vectorstore/  # Embeddings, vector store, retrieval
├── generation/   # LLM-backed answer generation (RAGAgent)
└── api/          # FastAPI service (upcoming)
```

Import everything via `rag_agent.*`, e.g. `from rag_agent.generation.rag_agent import RAGAgent`.

## Vector search

Two search backends, selected with `INDEX_TYPE` (see `.env.example`):

| `INDEX_TYPE` | How it works | When to use it |
|---|---|---|
| `brute-force` (default) | Exact cosine similarity against every stored vector. | Small corpora, and as the correctness baseline. |
| `ivf` | Inverted file index: a k-means (implemented from scratch) splits the vectors into `IVF_NLIST` lists; a query only scans the `NPROBE` lists whose centroids are closest to it. | Larger corpora, where scanning every vector gets slow. |

The index is persisted next to the store (`ivf_index.npz`) and reloaded on
startup; a missing or out-of-sync index is rebuilt automatically.

- **`IVF_NLIST`**: number of lists. `0` (default) picks about `sqrt(N)`.
- **`NPROBE`**: lists scanned per query, which trades speed for recall. Higher
  values give better recall but slower queries, and `NPROBE = IVF_NLIST` returns
  exactly the brute-force results. It can also be overridden per query with
  `VectorStore.search(..., nprobe=...)`.

### Incremental updates vs. rebuilds

Re-running k-means every time a document is ingested would cost a full
pass over the corpus. So newly-ingested chunks are instead **filed into the
list of their nearest existing centroid**, and the centroids stay put. This
is cheap, but the centroids were fitted on the corpus as it was at build
time:

- if new documents cover the **same topics**, the old centroids still fit
  them. Recall barely moves, but the lists grow, and so does query latency.
- if new documents cover **new topics**, the new vectors get crammed into
  lists centred elsewhere. Recall then drops quickly, because a query about the
  new topic probes lists that are not the ones holding its neighbours.

To bound this drift, the store counts the vectors added since the last
build and **rebuilds the index from scratch** (new k-means, `IVF_NLIST`
re-derived from the new size) once they exceed `IVF_REBUILD_THRESHOLD`
of the index (default `0.2`, i.e. 20%). A lower threshold keeps recall
closer to a fresh index at the cost of more frequent rebuilds; a higher
one rebuilds less often but tolerates more drift. `VectorStore.rebuild_index()`
forces a rebuild at any time.

### Benchmark: brute-force vs. IVF

Measured on **31,800 real chunks**: 40 public-domain books from Project
Gutenberg, split with the project's fixed-size chunker (500 characters)
and embedded with `all-MiniLM-L6-v2` (384 dimensions). The machine has a
single CPU core. Recall@10 is measured against the exact brute-force
top-10, over 200 held-out queries.

| Method | nprobe | Vectors scanned | Recall@10 | Median latency | Speed-up |
|---|---:|---:|---:|---:|---:|
| brute-force | — | 100% | 1.000 | 24.3 ms | 1x |
| IVF (178 lists) | 4 | 2.6% | 0.659 | 0.52 ms | 47x |
| IVF | 8 | 5.1% | 0.766 | 0.95 ms | 26x |
| IVF | 16 | 10.0% | 0.858 | 2.21 ms | 11x |
| IVF | 32 | 19.5% | 0.926 | 4.52 ms | 5.4x |
| IVF | 64 | 37.9% | 0.974 | 11.65 ms | 2.1x |

![IVF recall/latency trade-off](docs/benchmarks/ivf_recall_latency.png)

- `NPROBE` is the recall/latency knob. At this corpus size, `NPROBE=32`
  keeps over 92% of the exact neighbours while being about 5x faster than
  brute-force. The default `NPROBE=8` favours speed, at 0.77 recall.
- At a fixed `nprobe`, recall slowly decreases as the corpus grows, because
  the number of lists grows with it (about `sqrt(N)`). Larger corpora
  therefore need a larger `nprobe` for the same recall.
- Latencies include the float64 cosine computation over the scanned vectors
  (see `brute_force.py`), which both methods share.

Full results for 5k, 10k, 20k and 31.8k vectors are in
[`docs/benchmarks/ivf.md`](docs/benchmarks/ivf.md). To reproduce them, or
to run the benchmark on your own persisted store:

```bash
# any folder of .txt / .md files; it is chunked and embedded once into --store
python benchmarks/ivf_benchmark.py --corpus path/to/texts --store bench_store
python benchmarks/ivf_benchmark.py --store vector_db   # an existing store
```

The plot needs `matplotlib`, which is optional. Without it, only the
Markdown table is written.

## Roadmap

This project is under active development. The full plan — local LLM migration, IVF/HNSW/PQ vector search, a FastAPI service, an answer-quality evaluation harness, and an agentic retrieval loop — is tracked as [GitHub milestones and issues](../../issues).

## Setup

```bash
poetry install
```

_The FastAPI service (`poetry run uvicorn ...`) will be documented once the API milestone lands._

## License

[MIT](LICENSE)
