# Answer-quality evaluation

Fast retrieval proves nothing about the answers. `rag-agent eval` asks the
agent every question of a golden dataset and scores what comes back, so a
change to chunking, embeddings, the index, the prompt or the model can be
judged on numbers instead of on a few hand-picked questions.

```bash
poetry run rag-agent eval                          # rule-based metrics only
poetry run rag-agent eval --judge --label baseline # + the local LLM as a judge
poetry run rag-agent eval --judge-model models/other.gguf   # a separate judge model
poetry run rag-agent eval --compare eval/runs/*-baseline eval/runs/*-rerank
poetry run rag-agent eval --retrieval-only --rerank   # context only, no LLM needed
```

`--rewrite` and `--rerank` switch on query rewriting and cross-encoder
reranking for the run (see the README, "Retrieval pipeline"), whatever
`.env` says; the report records which steps were on.

`--retrieval-only` scores the context alone: it runs query search,
reranking and the relevance filter, but loads no LLM and generates nothing.
Only evidence recall, MRR, refusals for lack of relevant context and latency
are reported. It is deterministic, takes seconds, and runs on machines that
cannot hold the model, which makes it the right check for chunking,
embedding, index or reranking changes. It cannot be combined with
`--rewrite` or `--judge`, which need the LLM.

The corpus of the dataset is ingested into an **in-memory** store: the index
of `VECTOR_STORE_PATH` is neither read nor modified. Every question is asked
without conversation history. The other settings (`INDEX_TYPE`,
`CHUNK_SIZE`, `MIN_RELEVANCE_SCORE`, ...) come from the environment as usual
and are recorded in the report.

## The golden dataset

`eval/golden/v1/` documents **Lumen Rail**, a fictional regional rail
operator, in six Markdown files. The facts are invented on purpose: a model
cannot know them from training, so an answer not taken from the retrieved
context shows up as a hallucination instead of passing as general knowledge.

`questions.jsonl` holds 40 questions, each with a hand-written reference
answer and the exact passages (quotes) that support it:

| Category | Questions | What a good answer does |
|---|---:|---|
| `factual` | 24 | gives the fact of one passage |
| `multi_doc` | 6 | combines passages from several documents |
| `no_answer` | 5 | says the documents do not contain the answer |
| `ambiguous` | 5 | covers every reading of the question |

Expected sources are quotes, not chunk ids, so the dataset stays valid
whatever the chunking or the embedding model. `load_dataset()` checks every
quote against its document and computes a **fingerprint** of the corpus and
the questions. Any edit goes into a new version folder (`v2/`): two runs are
comparable only with the same fingerprint, and `--compare` refuses the others.

## Metrics

| Metric | Computed on | How |
|---|---|---|
| Evidence recall | answerable questions | share of expected quotes found in the chunks actually given to the LLM (after relevance filtering and prompt truncation) |
| MRR | answerable questions | 1 / rank of the first chunk holding an expected quote |
| Correct refusals | `no_answer` | share of unanswerable questions the agent declined |
| False refusals | answerable questions | share of answerable questions the agent declined |
| Faithfulness, rules | answers given | share of the answer's content words found in the context or the question |
| Answer relevance, rules | answerable questions | share of the reference answer's content words and numbers the answer contains |
| Hallucination rate, rules | all questions | an answerable question answered with a number absent from the context, or with less than half of its content words in it; or an unanswerable question answered anyway |
| Faithfulness, judge | answers given | the judge says every claim is SUPPORTED by the context |
| Answer correctness, judge | answerable questions | CORRECT = 1, PARTIAL = 0.5, INCORRECT or refusal = 0 against the reference |
| Hallucination rate, judge | all questions | an answer judged UNSUPPORTED, or an unanswerable question answered anyway |

A refusal is either the fixed message sent without calling the LLM (no
relevant chunk) or an answer worded as one ("the documents do not
mention...").

## Rules or LLM judge: the trade-off

Both are reported because neither is enough on its own.

**Rule-based checks** (`rag_agent/evaluation/metrics.py`) are deterministic,
free and instant: the same answers always get the same scores, so a
difference between two runs comes from the change under test, never from the
evaluator. They are also blind to meaning. A faithful paraphrase that uses
other words loses lexical support; an answer that copies context words into
a wrong statement keeps it. Invented numbers are the exception: on a corpus
of invented figures, a number absent from the context is the most reliable
hallucination signal, and the rules catch it every time.

**The LLM judge** (`rag_agent/evaluation/judge.py`) reads the answer like a
person would: it accepts paraphrases and catches a wrong claim made of the
right words. In exchange it costs one or two extra generations per question
(minutes on a CPU), and it is a model, so it errs too. With a small local
model judging its own answers it tends to be lenient. The judge only has to
reply with one closed keyword (SUPPORTED / UNSUPPORTED, CORRECT / PARTIAL /
INCORRECT) so that even a 3B model can do it and the verdict is parsed
without guessing; a reply without keyword is counted as **unparsed** in the
report rather than interpreted. `--judge-model` lets a larger or different
model grade the answers, which removes the self-grading bias.

In practice: run the rules on every change, add `--judge` before drawing a
conclusion, and read the per-question table whenever the two disagree.

## Reports

Each run writes `eval/runs/<date>-<label>/` (ignored by git):

- `report.json`: run metadata (dataset fingerprint, model, judge, `top_k`,
  every setting, machine), the summary and every question with its answer,
  sources, scores and judge reasons;
- `report.md`: the summary, a table per category and one line per question.

`--compare RUN RUN...` prints the summaries side by side as a Markdown
table, to paste in a pull request or the README.
