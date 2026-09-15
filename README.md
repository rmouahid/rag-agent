# RAG Agent

A minimal but precise Retrieval-Augmented Generation (RAG) agent built from scratch in Python — no LangChain, no managed vector database, no external ANN library.

## Features

- Multi-source ingestion (PDF, Markdown, Web)
- Custom chunking & embedding pipeline, built without a framework
- Brute-force cosine similarity vector search today, with an IVF index (also built from scratch) in progress
- Answer generation currently via the Groq API (LLaMA 3.3 70B) as a working prototype, being migrated to a fully local LLM runtime — see the roadmap below

## Stack

- Python 3.11+
- Groq API (LLaMA 3.3 70B) — prototype generation backend, being replaced by a local LLM runtime
- sentence-transformers (embeddings)
- numpy (vector search)

## Roadmap

This project is under active development. The full plan — local LLM migration, IVF/HNSW/PQ vector search, a FastAPI service, an answer-quality evaluation harness, and an agentic retrieval loop — is tracked as [GitHub milestones and issues](../../issues).

## Setup

_Coming soon — will land once dependency management (Poetry) and the API milestone are complete._
