"""
Évaluation de la recherche sur un jeu public BEIR (par défaut SciFact : 5 183
résumés scientifiques, 300 questions de test annotées par des experts).

Contrairement au jeu doré du projet (eval/golden/), les annotations ne sont
pas les nôtres : les chiffres sont comparables à ceux publiés pour le même
modèle d'embedding. Aucun LLM n'est utilisé.

Configurations mesurées, toutes avec l'embedder du projet (EMBEDDING_MODEL) :
- document         : un vecteur par document (titre + texte), comme les
                     évaluations BEIR / MTEB, scan exact
- chunks           : le chunker du projet (fixe, 500 caractères, recouvrement
                     50) ; un document prend le score de son meilleur chunk
- chunks + HNSW    : mêmes chunks, recherche approchée par notre index HNSW
- chunks + rerank  : 20 meilleurs chunks rescorés par le cross-encoder

Métriques au niveau document, sur les 10 premiers : nDCG@10 (la métrique de
référence de BEIR), MRR@10 et hit@k (au moins un document pertinent dans les
k premiers) pour k = 1, 5, 10. Latence : temps de recherche par question
(embedding de la question compris, un thread), p50 et p95.

Usage :
    python benchmarks/beir_eval.py --data beir/scifact
    python benchmarks/beir_eval.py --data beir/scifact --no-rerank

`--data` est le dossier dézippé de
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
(corpus.jsonl, queries.jsonl, qrels/test.tsv). Les embeddings sont gardés dans
`--work` pour les relances. Écrit docs/benchmarks/beir_<dataset>.md.
"""

import argparse
import csv
import json
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rag_agent.chunking.factory import get_chunker  # noqa: E402
from rag_agent.ingestion.document import Document  # noqa: E402
from rag_agent.vectorstore.brute_force import cosine_top_k  # noqa: E402
from rag_agent.vectorstore.hnsw import HNSWIndex  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "benchmarks")
K = 10
HITS = (1, 5, 10)


def load_beir(path: str, split: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """corpus {doc_id: titre + texte}, questions {qid: texte}, qrels {qid: {doc_id: pertinence}}"""
    corpus = {}
    with open(os.path.join(path, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[str(doc["_id"])] = f"{doc.get('title', '')}\n{doc['text']}".strip()
    qrels: dict[str, dict[str, int]] = {}
    with open(os.path.join(path, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = {}
    with open(os.path.join(path, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            query = json.loads(line)
            if str(query["_id"]) in qrels:
                queries[str(query["_id"])] = query["text"]
    return corpus, queries, qrels


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranked[:k]) if d in relevant), 0.0)


def unique_docs(chunk_ids, chunk_doc: list[str]) -> list[str]:
    """Documents dans l'ordre de leur meilleur chunk (score max par document)"""
    seen, docs = set(), []
    for i in chunk_ids:
        doc = chunk_doc[i]
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def embedded(embedder, texts: list[str], path: str) -> np.ndarray:
    if os.path.isfile(path):
        vectors = np.load(path, allow_pickle=False)
        if len(vectors) == len(texts):
            return vectors
    print(f"Embedding {len(texts):,} texts -> {path}", flush=True)
    vectors = embedder.embed_many(texts).astype(np.float32)
    np.save(path, vectors)
    return vectors


def evaluate(name: str, search, queries: dict[str, str], qrels, embedder) -> dict:
    """`search(text, vector) -> doc ids classés` ; latence embedding compris"""
    scores = {"ndcg": [], "mrr": [], **{f"hit@{k}": [] for k in HITS}}
    times = []
    for qid, text in queries.items():
        start = time.perf_counter()
        ranked = search(text, embedder.embed(text))
        times.append(time.perf_counter() - start)
        relevant = qrels[qid]
        scores["ndcg"].append(ndcg_at_k(ranked, relevant))
        scores["mrr"].append(mrr_at_k(ranked, relevant))
        for k in HITS:
            scores[f"hit@{k}"].append(float(any(d in relevant for d in ranked[:k])))
    row = {"config": name, **{m: float(np.mean(v)) for m, v in scores.items()}}
    row["p50_ms"] = 1000 * float(np.median(times))
    row["p95_ms"] = 1000 * float(np.percentile(times, 95))
    print(f"  {name:<26} nDCG@10={row['ndcg']:.3f} MRR@10={row['mrr']:.3f} p50={row['p50_ms']:.1f} ms", flush=True)
    return row


def to_markdown(rows: list[dict], args, stats: dict) -> str:
    lines = [
        f"# Retrieval on BEIR {args.dataset}",
        "",
        "_Generated by `benchmarks/beir_eval.py`; do not edit by hand._",
        "",
        f"- Data: BEIR {args.dataset}, {stats['docs']:,} documents ({stats['chunks']:,} chunks with the"
        f" project's fixed-size chunker), {stats['queries']} `{args.split}` questions with expert relevance labels",
        f"- Embeddings: `{stats['model']}`; reranker: `{stats['reranker'] or 'not run'}`",
        f"- Metrics at document level, top {K}: nDCG@{K}, MRR@{K}, hit@k = share of questions with at least"
        " one relevant document in the top k",
        "- Latency: per question, question embedding included, one thread",
        f"- Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} CPU,"
        f" Python {platform.python_version()}, numpy {np.__version__}",
        "",
        f"| Configuration | nDCG@{K} | MRR@{K} | hit@1 | hit@5 | hit@10 | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['ndcg']:.3f} | {r['mrr']:.3f} | {r['hit@1']:.1%} | {r['hit@5']:.1%}"
            f" | {r['hit@10']:.1%} | {r['p50_ms']:.1f} ms | {r['p95_ms']:.1f} ms |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="unzipped BEIR dataset folder")
    parser.add_argument("--dataset", help="name in the report (default: folder name)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--work", default="beir_work", help="folder for cached embeddings")
    parser.add_argument("--ef", type=int, default=64, help="HNSW ef for the approximate configuration")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder configuration")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.dataset = args.dataset or os.path.basename(os.path.normpath(args.data))

    from rag_agent.vectorstore.embedder import Embedder

    corpus, queries, qrels = load_beir(args.data, args.split)
    doc_ids = list(corpus)
    chunker = get_chunker(strategy="fixed", chunk_size=500, overlap=50)
    chunks, chunk_doc = [], []
    for doc_id in doc_ids:
        for chunk in chunker.chunk(Document(text=corpus[doc_id], source=doc_id, metadata={"type": "text"})):
            chunks.append(chunk.text)
            chunk_doc.append(doc_id)
    print(f"{len(doc_ids):,} documents, {len(chunks):,} chunks, {len(queries)} questions", flush=True)

    os.makedirs(args.work, exist_ok=True)
    embedder = Embedder(cache_dir="")
    doc_vectors = embedded(embedder, [corpus[d] for d in doc_ids], os.path.join(args.work, f"{args.dataset}_docs.npy"))
    chunk_vectors = embedded(embedder, chunks, os.path.join(args.work, f"{args.dataset}_chunks.npy"))
    depth = 10 * K  # chunks lus pour remplir 10 documents distincts

    rows = [
        evaluate(
            "document, exact",
            lambda text, q: [doc_ids[i] for i in cosine_top_k(doc_vectors, q, K)[0]],
            queries,
            qrels,
            embedder,
        ),
        evaluate(
            "chunks, exact",
            lambda text, q: unique_docs(cosine_top_k(chunk_vectors, q, depth)[0], chunk_doc),
            queries,
            qrels,
            embedder,
        ),
    ]

    print(f"Building HNSW over {len(chunks):,} chunks ...", flush=True)
    hnsw = HNSWIndex.build(chunk_vectors)
    rows.append(
        evaluate(
            f"chunks, HNSW ef={args.ef}",
            lambda text, q: unique_docs(hnsw.search(chunk_vectors, q, depth, max(args.ef, depth))[0], chunk_doc),
            queries,
            qrels,
            embedder,
        )
    )

    reranker_name = None
    if not args.no_rerank:
        from rag_agent.vectorstore.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker_name = reranker.model_name

        def rerank(text, q):
            ids, _ = cosine_top_k(chunk_vectors, q, depth)
            head = [{"id": int(i), "text": chunks[i]} for i in ids[: args.rerank_candidates]]
            order = [c["id"] for c in reranker.rerank(text, head)] + [int(i) for i in ids[args.rerank_candidates :]]
            return unique_docs(order, chunk_doc)

        rows.append(evaluate(f"chunks, exact + rerank top {args.rerank_candidates}", rerank, queries, qrels, embedder))

    stats = {
        "docs": len(doc_ids),
        "chunks": len(chunks),
        "queries": len(queries),
        "model": embedder.model_name,
        "reranker": reranker_name,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    report = to_markdown(rows, args, stats)
    with open(os.path.join(args.output_dir, f"beir_{args.dataset}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
