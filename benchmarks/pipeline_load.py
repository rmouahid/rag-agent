"""Measure the real ingestion-to-generation pipeline on a supplied corpus.

Run from the repository root: python -m benchmarks.pipeline_load --help.
No stub embeddings, synthetic answers or pre-populated vector index are used.
"""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_agent.config import config
from rag_agent.evaluation.harness import agent_options
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.pipeline import FILE_TYPES
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.vectorstore.vector_store import VectorStore


def peak_rss_mb() -> float:
    """Peak resident memory of this process, including native LLM allocations."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024 if sys.platform == "darwin" else 1024)


def latency_summary(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {}
    return {
        "p50_seconds": float(np.percentile(seconds, 50)),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
    }


def load_questions(path: Path) -> list[str]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        question = item.get("question") if isinstance(item, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{line_number}: expected a non-empty question string")
        questions.append(question.strip())
    if not questions:
        raise ValueError("the question set is empty")
    return questions


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    if args.min_chunks < 1 or args.repeats < 1 or args.top_k < 1:
        raise ValueError("min-chunks, repeats and top-k must be positive")
    questions = load_questions(args.questions)
    paths = sorted(p for p in args.corpus.rglob("*") if p.is_file() and p.suffix.lower() in FILE_TYPES)
    if not paths:
        raise ValueError("the corpus contains no supported documents")
    if args.output.exists():
        raise ValueError(f"{args.output} exists; use a new output path to preserve the previous run")

    report = {
        "status": "running",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "settings": asdict(config),
        "minimum_chunks": args.min_chunks,
        "corpus_files": [],
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "question_count": len(questions),
        "repeats": args.repeats,
        "top_k": args.top_k,
        "queries": [],
    }
    start = time.perf_counter()
    write_report(args.output, report)
    try:
        # The persistent application index is never read or modified.
        agent = RAGAgent(
            retriever=Retriever(vector_store=VectorStore(path=None)),
            history_turns=0,
            agentic=not args.no_agentic,
        )
        report["agent"] = agent_options(agent)
        report["initialization_seconds"] = time.perf_counter() - start
        ingestion_start = time.perf_counter()
        for path in paths:
            before = time.perf_counter()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result = agent.pipeline.ingest(str(path), source_name=path.relative_to(args.corpus).as_posix())
            report["corpus_files"].append(
                {
                    "path": path.relative_to(args.corpus).as_posix(),
                    "sha256": digest.hexdigest(),
                    "chunks_total": result.chunks_total,
                    "chunks_indexed": result.chunks_indexed,
                    "seconds": time.perf_counter() - before,
                }
            )
            report["indexed_chunks"] = len(agent.retriever.vector_store)
            report["peak_rss_mb"] = peak_rss_mb()
            write_report(args.output, report)
            print(f"{path.name}: {report['indexed_chunks']} indexed chunks", flush=True)

        report["ingestion_seconds"] = time.perf_counter() - ingestion_start
        if report["indexed_chunks"] < args.min_chunks:
            raise ValueError(
                f"only {report['indexed_chunks']} distinct chunks after deduplication; "
                f"the required scale is {args.min_chunks}"
            )
        report["peak_rss_after_ingestion_mb"] = peak_rss_mb()
        # First request is included and identified: no hidden warm-up or dropping
        # failed/refused queries from the latency distribution.
        for repeat in range(args.repeats):
            for number, question in enumerate(questions):
                before = time.perf_counter()
                answer = agent.ask(question, top_k=args.top_k, history=[])
                elapsed = time.perf_counter() - before
                report["queries"].append(
                    {
                        "question_index": number,
                        "repeat": repeat,
                        "seconds": elapsed,
                        "status": answer.status,
                        "search_count": len(answer.steps),
                        "answer_characters": len(answer.text),
                        "sources": len(answer.sources),
                    }
                )
                report["peak_rss_mb"] = peak_rss_mb()
                write_report(args.output, report)
                print(f"request {len(report['queries'])}: {elapsed:.2f}s ({answer.status})", flush=True)
        report["latency"] = latency_summary([row["seconds"] for row in report["queries"]])
        report["answered_requests"] = sum(row["status"] == "answered" for row in report["queries"])
        if report["answered_requests"] == 0:
            raise ValueError("no request reached generation: this is not a full-pipeline measurement")
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - start
        report["peak_rss_mb"] = peak_rss_mb()
        report["finished_at"] = datetime.now(tz=UTC).isoformat()
        write_report(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True, help="JSONL: one question string per object")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-chunks", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-agentic", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "latency": report["latency"], "peak_rss_mb": report["peak_rss_mb"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
