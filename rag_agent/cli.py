"""
Ligne de commande, pour un usage local sans lancer l'API :

    rag-agent ingest <fichier|dossier|url>...   indexe des sources
    rag-agent ask "<question>"                  répond à partir de l'index
    rag-agent eval [--judge]                    note les réponses sur le jeu de référence
    rag-agent eval --compare RUN RUN...         compare des runs d'évaluation

L'index est celui de VECTOR_STORE_PATH, partagé avec l'API et `main.py` :
ce qu'`ingest` y ajoute est disponible pour `ask` et pour le service.
`eval` travaille sur un index en mémoire, sans toucher à celui-ci.
"""

import argparse
import os
import sys
from collections.abc import Iterator
from typing import cast

from rag_agent.config import config
from rag_agent.evaluation import harness
from rag_agent.evaluation.dataset import DatasetError, load_dataset
from rag_agent.evaluation.judge import LLMJudge
from rag_agent.generation.llm import LocalLLM, ModelNotFoundError
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.generation.truncation import ContextOverflowError
from rag_agent.pipeline import FILE_TYPES
from rag_agent.vectorstore.reranker import CrossEncoderReranker
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.vectorstore.vector_store import VectorStore

DEFAULT_DATASET = os.path.join("eval", "golden", "v1")
DEFAULT_RUNS_DIR = os.path.join("eval", "runs")


def _files_in(folder: str) -> list[str]:
    """
    Fichiers d'un type pris en charge sous `folder`, récursivement, triés
    """
    found = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        found += [os.path.join(root, name) for name in sorted(files) if name.lower().endswith(tuple(FILE_TYPES))]
    return found


def _sources(paths: list[str]) -> Iterator[str | None]:
    """
    Les URL et fichiers tels quels, les dossiers remplacés par leurs
    fichiers ; None pour un dossier qui n'en contient aucun (signalé)
    """
    for path in paths:
        if not os.path.isdir(path):
            yield path
            continue
        files = _files_in(path)
        if not files:
            print(f"{path}: no {', '.join(FILE_TYPES)} file found", file=sys.stderr)
            yield None
        yield from files


def ingest(args: argparse.Namespace) -> int:
    # Le LLM n'est pas chargé : l'ingestion n'a besoin que des embeddings
    retriever = Retriever()
    if not retriever.vector_store.path:
        print("VECTOR_STORE_PATH is empty: ingested chunks would be lost on exit. Set it first.", file=sys.stderr)
        return 1

    failures = 0
    for source in _sources(args.sources):
        if source is None:
            failures += 1
            continue
        if not source.startswith(("http://", "https://")) and not os.path.exists(source):
            print(f"{source}: no such file or directory", file=sys.stderr)
            failures += 1
            continue
        try:
            result = retriever.pipeline.ingest(source)
        except Exception as e:  # une source en échec n'empêche pas les suivantes
            print(f"{source}: {e}", file=sys.stderr)
            failures += 1
            continue
        print(f"{source}: {result.chunks_indexed} chunks indexed ({result.duplicates} duplicates skipped)")

    print(f"{len(retriever.vector_store)} chunks in {retriever.vector_store.path}")
    return 1 if failures else 0


def ask(args: argparse.Namespace) -> int:
    try:
        agent = RAGAgent()
    except ModelNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        answer = agent.ask_stream(args.question, top_k=args.top_k)
    except ContextOverflowError as e:
        print(e, file=sys.stderr)
        return 1

    first = True
    for piece in answer:
        print(piece.lstrip() if first else piece, end="", flush=True)
        first = False
    print()
    if answer.sources:
        print(f"\nSources:\n{answer.format_sources()}")
    return 0


class NoLLM:
    """LLM absent d'une évaluation --retrieval-only : rien ne doit l'appeler"""

    model_path = "none"

    def generate(self, prompt: str) -> str:
        raise RuntimeError("no LLM is loaded in --retrieval-only mode")


def evaluate(args: argparse.Namespace) -> int:
    if args.compare:
        try:
            print(harness.compare([harness.load_report(path) for path in args.compare]), end="")
        except (OSError, ValueError) as e:
            print(e, file=sys.stderr)
            return 1
        return 0

    try:
        dataset = load_dataset(args.dataset)
    except (OSError, DatasetError) as e:
        print(f"{args.dataset}: {e}", file=sys.stderr)
        return 1
    if args.retrieval_only and (args.rewrite or args.judge or args.judge_model):
        print("--retrieval-only runs without the LLM: it cannot rewrite queries or judge answers", file=sys.stderr)
        return 1
    try:
        # Sans génération, l'agent n'appelle jamais son LLM : NoLLM le garantit
        llm = cast(LocalLLM, NoLLM()) if args.retrieval_only else LocalLLM()
        judge_llm = LocalLLM(model_path=args.judge_model) if args.judge_model else llm
    except ModelNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    agent = RAGAgent(
        llm=llm,
        history_turns=0,
        retriever=Retriever(vector_store=VectorStore(path=None)),
        query_rewriting=False if args.retrieval_only else args.rewrite or None,
        reranker=CrossEncoderReranker() if args.rerank else None,
    )
    n_chunks = harness.ingest_corpus(agent, dataset)
    print(f"{dataset.name} v{dataset.version}: {len(dataset.questions)} questions, {n_chunks} chunks indexed")

    def progress(i: int, result: harness.QuestionResult) -> None:
        if result.retrieval_only:
            flag = "no relevant context" if result.refused else f"evidence recall {result.evidence_recall:.0%}"
        else:
            flag = "refused" if result.refused else ("HALLUCINATION?" if result.hallucination_rules else "ok")
        print(f"[{i}/{len(dataset.questions)}] {result.id} {result.category}: {flag} ({result.seconds:.1f} s)")

    judge = LLMJudge(judge_llm) if args.judge or args.judge_model else None
    results = harness.evaluate(
        agent, dataset, top_k=args.top_k, judge=judge, progress=progress, retrieval_only=args.retrieval_only
    )
    summary = harness.summarize(results)
    meta = harness.run_metadata(
        dataset, args.label, args.top_k, llm.model_path, judge_llm.model_path if judge else None
    )
    meta["agent"] = harness.agent_options(agent, args.retrieval_only)
    run_dir = harness.write_report(args.output_dir, meta, summary, results)
    print(f"\nReport written to {run_dir}")
    with open(os.path.join(run_dir, harness.REPORT_MD), encoding="utf-8") as f:
        print("".join(f.readlines()[:30]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-agent",
        description="Index documents and ask questions about them with a local LLM.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser(
        "ingest",
        help="index files, folders or web pages",
        description=f"Index sources into the vector store at VECTOR_STORE_PATH ({config.vector_store_path!r}). "
        f"Folders are searched recursively for {', '.join(FILE_TYPES)} files.",
    )
    ingest_parser.add_argument("sources", nargs="+", metavar="SOURCE", help="file, folder or http(s) URL")
    ingest_parser.set_defaults(handler=ingest)

    ask_parser = commands.add_parser(
        "ask",
        help="answer a question from the indexed documents",
        description="Answer a question from the indexed documents, streaming the answer, then list its sources.",
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5, help="chunks retrieved before relevance filtering")
    ask_parser.set_defaults(handler=ask)

    eval_parser = commands.add_parser(
        "eval",
        help="score the answers on a golden question set",
        description="Ingest a golden dataset into an in-memory index, ask every question and score the answers "
        "(retrieval, refusals, faithfulness, relevance, hallucinations). See docs/evaluation.md.",
    )
    eval_parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"dataset folder (default: {DEFAULT_DATASET})")
    eval_parser.add_argument("--top-k", type=int, default=5, help="chunks retrieved before relevance filtering")
    eval_parser.add_argument("--judge", action="store_true", help="also grade answers with the LLM as a judge")
    eval_parser.add_argument("--judge-model", help="GGUF weights of a separate judge model (implies --judge)")
    eval_parser.add_argument(
        "--rewrite", action="store_true", help="rewrite questions into search queries (default: QUERY_REWRITING)"
    )
    eval_parser.add_argument(
        "--rerank", action="store_true", help="rerank candidates with the cross-encoder (default: RERANK)"
    )
    eval_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score the retrieved context only, without loading the LLM or generating answers",
    )
    eval_parser.add_argument("--label", default="run", help="name of the run, used in the report folder")
    eval_parser.add_argument("--output-dir", default=DEFAULT_RUNS_DIR, help=f"default: {DEFAULT_RUNS_DIR}")
    eval_parser.add_argument("--compare", nargs="+", metavar="RUN", help="compare existing runs instead")
    eval_parser.set_defaults(handler=evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


def run() -> None:
    """Point d'entrée du script `rag-agent` (pyproject.toml)."""
    sys.exit(main())
