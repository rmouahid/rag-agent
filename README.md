# RAG Agent

A minimal but precise Retrieval-Augmented Generation (RAG) agent built from scratch in Python — no LangChain, no managed vector database, no external ANN library.

## Features

- Multi-source ingestion (PDF, Markdown, Web)
- Custom chunking & embedding pipeline, built without a framework
- Brute-force cosine similarity vector search today, with an IVF index (also built from scratch) in progress
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

## Roadmap

This project is under active development. The full plan — local LLM migration, IVF/HNSW/PQ vector search, a FastAPI service, an answer-quality evaluation harness, and an agentic retrieval loop — is tracked as [GitHub milestones and issues](../../issues).

## Setup

```bash
poetry install
```

_The FastAPI service (`poetry run uvicorn ...`) will be documented once the API milestone lands._

## License

[MIT](LICENSE)
