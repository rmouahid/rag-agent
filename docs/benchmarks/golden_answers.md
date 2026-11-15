# Generated-answer evaluation on Lumen Rail v2

Measured on 2026-11-15 with the unmodified evaluation harness, not with a
stub LLM. The versioned corpus is `eval/golden/v2/`: six fictional Lumen Rail
documents, 19 chunks with fixed 500-character chunking/50-character overlap,
and 50 questions (28 factual, 9 multi-document, 7 unanswerable, 6 ambiguous).
Dataset fingerprint: `2f438011f572`. Top-k is 5 and the relevance threshold
is 0.25. Query rewriting and reranking are disabled.

## Baseline result

| Metric | TinyLlama baseline |
|---|---:|
| Questions | 50 |
| Evidence recall | 96.1% |
| MRR | 0.833 |
| Faithfulness, lexical rules | 28.1% |
| Answer relevance, lexical rules | 23.9% |
| Hallucination flags, rules | 92.0% |
| Correct refusal on unanswerable questions | 0.0% |
| False refusal on answerable questions | 0.0% |
| Median full-answer latency | 19.9 s |
| Mean full-answer latency | 24.1 s |

The [complete baseline report](runs/tinyllama-baseline.json) contains all
50 answers, their supplied passages, per-question timing and rule scores.
The model is **TinyLlama 1.1B Chat v1.0, Q4_K_M**, not the project's default
Llama 3.2 3B. This is a negative result for that small model/configuration:
strong passage retrieval did not produce reliable generated answers. All
seven unanswerable questions were answered instead of being declined.

Rules are lexical proxies, not human or LLM-judge verdicts. They penalise
paraphrases and language changes; a flag is not a proved hallucination.
There is no judge run in these results. Do not read 28.1% as the share of
factually correct answers, or extrapolate the result to other models.

## Runtime and reproduction

- GGUF SHA-256:
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
- Context 2048, maximum response 192 tokens, temperature 0, four LLM
  threads, 64-token context margin; exact cosine search.
- Embeddings: cached `all-MiniLM-L6-v2`; no generated or stub embeddings.
- x86-64, 12 logical CPUs, Python 3.12.3; installed runtime:
  llama-cpp-python 0.3.35, sentence-transformers 5.7.0, torch 2.7.1,
  numpy 2.1.3. This existing environment differs from the Poetry lock;
  matching these versions is required for a close reproduction.
- Runs share CPU resources with other work. Latencies describe this
  measurement environment, not an isolated performance claim.

From the repository root, with those dependencies and cached model weights:

```bash
export LOCAL_MODEL_PATH=/path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
export LLM_CONTEXT_SIZE=2048 LLM_MAX_TOKENS=192 LLM_TEMPERATURE=0 LLM_THREADS=4
export QUERY_REWRITING=false RERANK=false AGENTIC_LOOP=false
poetry run rag-agent eval --label tinyllama-baseline
poetry run rag-agent eval --agentic --label tinyllama-agentic
```

The agentic comparison is reported separately after its full 50-question
run has finished. A smoke run or an unfinished log is not a comparison.
This corpus is small: the figures do not satisfy the 200k-chunk load test.
