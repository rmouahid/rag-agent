# ADR 0001: Local LLM runtime choice

## Status

Accepted

## Context

The project currently generates answers via Groq's hosted API
(`rag_agent/generation/rag_agent.py`), which contradicts the stated
target architecture: no external inference API, a locally-hosted LLM
only (see README). The migration itself is tracked by #25; this ADR
only decides *which* local runtime to integrate.

The development machine is CPU-only (no GPU available), which rules
out anything that assumes CUDA by default. The runtime also needs to
support token-by-token streaming (#28) and package cleanly as a
regular Python dependency, consistent with the Poetry-based setup
from #3 — no separate service/daemon to install and keep running.

Three options were compared:

| | Memory footprint | Streaming | Packaging |
|---|---|---|---|
| **llama.cpp** (`llama-cpp-python`) | Best — GGUF quantized weights (Q4_K_M ≈ 2-5 GB for 3B-8B models) | Native, token-by-token via callback/generator | Pure Python dependency (prebuilt wheels for common platforms), no external process |
| **Ollama** | Good — same GGUF quantization under the hood | Yes, via its HTTP API | Requires a separate background daemon (`ollama serve`) installed on the host, outside the Python packaging story |
| **transformers pipeline** | Worst on CPU — fp16/fp32 weights (7B ≈ 14 GB+ RAM) unless paired with `bitsandbytes`, which is GPU-oriented and has patchy CPU/platform support | Supported (`TextIteratorStreamer`) but more boilerplate | Heaviest dependency set (`transformers` + `torch` + `accelerate`), primarily optimized for GPU; slow on CPU |

## Decision

Use **llama.cpp via the `llama-cpp-python` bindings**.

It is the only option that is both a pure Python/Poetry dependency
(no external daemon, unlike Ollama) and efficient on CPU-only hardware
via GGUF quantization (unlike a plain `transformers` pipeline, which
targets GPU inference). It also provides native streaming, which #28
requires.

### Pinned model

- **Default**: `Llama-3.2-3B-Instruct`, `Q4_K_M` GGUF quantization —
  chosen for responsive CPU latency on modest hardware (~2 GB
  resident).
- **Optional upgrade path**: `Llama-3.1-8B-Instruct`, `Q4_K_M` GGUF,
  for better answer quality on machines with more RAM/CPU headroom.

Both are swappable via `LOCAL_MODEL_PATH` in the central config
(`rag_agent/config.py`, #4) — no code change needed to switch.

## Consequences

- Adds `llama-cpp-python` as a runtime dependency (to be added when
  #25 implements the integration, not by this ADR).
- Requires downloading and storing GGUF model weights locally
  (~2-5 GB depending on the model), on first run.
- CPU inference latency will be noticeably higher than Groq's hosted,
  GPU-backed API — acceptable trade-off for the "no external
  inference API" requirement.
- No hosted fine-tuning/model-management API, unlike Ollama's model
  registry — model files are managed manually.
