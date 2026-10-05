"""
Ligne de commande, pour un usage local sans lancer l'API :

    rag-agent ingest <fichier|dossier|url>...   indexe des sources
    rag-agent ask "<question>"                  répond à partir de l'index

L'index est celui de VECTOR_STORE_PATH, partagé avec l'API et `main.py` :
ce qu'`ingest` y ajoute est disponible pour `ask` et pour le service.
"""

import argparse
import os
import sys
from collections.abc import Iterator

from rag_agent.config import config
from rag_agent.generation.llm import ModelNotFoundError
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.generation.truncation import ContextOverflowError
from rag_agent.pipeline import FILE_TYPES
from rag_agent.vectorstore.retriever import Retriever


def _sources(paths: list[str]) -> Iterator[str]:
    """
    Les URL et fichiers tels quels ; les dossiers, parcourus récursivement,
    donnent leurs fichiers d'un type pris en charge, triés
    """
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for name in sorted(files):
                    if name.lower().endswith(tuple(FILE_TYPES)):
                        yield os.path.join(root, name)
        else:
            yield path


def ingest(args: argparse.Namespace) -> int:
    # Le LLM n'est pas chargé : l'ingestion n'a besoin que des embeddings
    retriever = Retriever()
    if not retriever.vector_store.path:
        print("VECTOR_STORE_PATH is empty: ingested chunks would be lost on exit. Set it first.", file=sys.stderr)
        return 1

    failures = 0
    for source in _sources(args.sources):
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


def run():
    """Point d'entrée du script `rag-agent` (pyproject.toml)."""
    sys.exit(main())
