"""
Benchmark brute-force vs IVF : recall@k et latence de requête.

Pour chaque taille de corpus, construit un index IVF (~sqrt(N) listes),
puis mesure pour une plage de `nprobe` :
- recall@k : part des k voisins exacts (brute-force) retrouvés par l'IVF
- latence  : temps médian et p95 d'une requête (une requête à la fois,
             comme dans le pipeline RAG)

Les vecteurs sont de vrais embeddings, lus dans un VectorStore persisté
(`--store`). Les requêtes sont des chunks du store tirés au hasard et
retirés du corpus ; les tailles plus petites sont des sous-échantillons.
`--corpus` construit d'abord ce store à partir d'un dossier de fichiers
texte (.txt / .md), avec le chunker et l'Embedder du projet.

Usage :
    python benchmarks/ivf_benchmark.py --corpus books/ --store bench_store --limit 32000
    python benchmarks/ivf_benchmark.py --store vector_db --sizes 5000 10000

Écrit docs/benchmarks/ivf.md et, si matplotlib est installé (optionnel),
docs/benchmarks/ivf_recall_latency.png.
"""

import argparse
import glob
import json
import os
import platform
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rag_agent.vectorstore.brute_force import cosine_top_k  # noqa: E402
from rag_agent.vectorstore.ivf import IVFIndex  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "benchmarks")


def build_store(corpus_dir: str, store_path: str, limit: int, seed: int) -> None:
    """
    Découpe les fichiers .txt/.md de `corpus_dir` (chunker "fixed", 500
    caractères, recouvrement 50), tire au plus `limit` chunks au hasard et
    les encode dans un VectorStore persisté à `store_path`. Les en-têtes et
    pieds de page du Projet Gutenberg sont retirés s'ils sont présents.
    """
    from rag_agent.chunking.factory import get_chunker
    from rag_agent.ingestion.document import Document
    from rag_agent.vectorstore.embedder import Embedder
    from rag_agent.vectorstore.vector_store import VectorStore

    chunker = get_chunker(strategy="fixed", chunk_size=500, overlap=50)
    chunks = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt")) + glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
        end = re.search(r"\*\*\* ?END OF", text)
        text = text[start.end() if start else 0 : end.start() if end else len(text)]
        chunks += chunker.chunk(Document(text=text, source=os.path.basename(path), metadata={"type": "text"}))
    if not chunks:
        raise SystemExit(f"no .txt/.md content found in {corpus_dir}")

    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(len(chunks), size=min(limit, len(chunks)), replace=False))
    chunks = [chunks[i] for i in keep]
    print(f"Embedding {len(chunks):,} chunks into {store_path} ...", flush=True)

    embedder = Embedder(cache_dir="")
    store = VectorStore(index_type="brute-force")
    for i in range(0, len(chunks), 1024):
        batch = chunks[i : i + 1024]
        store.add(batch, embedder.embed_many([c.text for c in batch]))
        print(f"  {i + len(batch):,}/{len(chunks):,}", flush=True)
    store.save(store_path)


def load_store(path: str, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    vectors = np.load(os.path.join(path, "vectors.npy"), allow_pickle=False)
    with open(os.path.join(path, "chunks.json"), encoding="utf-8") as f:
        sources = {c["source"] for c in json.load(f)}
    if len(vectors) <= n_queries:
        raise SystemExit(f"{path} holds {len(vectors)} vectors: not enough for {n_queries} held-out queries")
    order = np.random.default_rng(seed).permutation(len(vectors))
    source = f"{len(vectors):,} chunks from {len(sources)} documents, {vectors.shape[1]}-dimensional embeddings"
    return vectors[order[n_queries:]], vectors[order[:n_queries]], source


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def run(corpus: np.ndarray, queries: np.ndarray, k: int, nprobes: list[int], seed: int) -> list[dict]:
    exact, exact_times = [], []
    for q in queries:
        (ids, _), elapsed = timed(cosine_top_k, corpus, q, k)
        exact.append(set(ids.tolist()))
        exact_times.append(elapsed)

    index, build_time = timed(IVFIndex.build, corpus, None, seed)
    rows = [{
        "n": len(corpus), "method": "brute-force", "n_lists": None, "nprobe": None,
        "recall": 1.0, "p50_ms": 1000 * np.median(exact_times), "p95_ms": 1000 * np.percentile(exact_times, 95),
        "scanned": 1.0, "build_s": None,
    }]

    for nprobe in [p for p in nprobes if p <= index.n_lists]:
        recalls, times, scanned = [], [], []
        for q, truth in zip(queries, exact):
            (ids, _), elapsed = timed(index.search, corpus, q, k, nprobe)
            recalls.append(len(truth & set(ids.tolist())) / len(truth))
            times.append(elapsed)
            scanned.append(sum(len(index.lists[i]) for i in index.probe(q, nprobe)) / len(corpus))
        rows.append({
            "n": len(corpus), "method": "ivf", "n_lists": index.n_lists, "nprobe": nprobe,
            "recall": float(np.mean(recalls)), "p50_ms": 1000 * np.median(times),
            "p95_ms": 1000 * np.percentile(times, 95), "scanned": float(np.mean(scanned)), "build_s": build_time,
        })
    return rows


def to_markdown(rows: list[dict], args, source: str) -> str:
    lines = [
        "# Brute-force vs IVF benchmark",
        "",
        "_Generated by `benchmarks/ivf_benchmark.py`; do not edit by hand._",
        "",
        f"- Data: {source}",
        f"- {args.queries} queries, recall@{args.k} measured against the exact brute-force top-{args.k}",
        "- Latency: one query at a time, median (p50) and 95th percentile (p95)",
        f"- Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} CPU, "
        f"Python {platform.python_version()}, numpy {np.__version__}",
        "",
    ]
    for n in sorted({r["n"] for r in rows}):
        group = [r for r in rows if r["n"] == n]
        ivf = [r for r in group if r["method"] == "ivf"]
        base = group[0]
        lines += [
            f"## N = {n:,} vectors",
            "",
            f"IVF: {ivf[0]['n_lists']} lists, built in {ivf[0]['build_s']:.1f} s." if ivf else "",
            "",
            f"| Method | nprobe | Scanned | Recall@{args.k} | p50 latency | p95 latency | Speed-up (p50) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in group:
            nprobe = "—" if r["nprobe"] is None else r["nprobe"]
            lines.append(
                f"| {r['method']} | {nprobe} | {100 * r['scanned']:.1f}% | {r['recall']:.3f} "
                f"| {r['p50_ms']:.2f} ms | {r['p95_ms']:.2f} ms | {base['p50_ms'] / r['p50_ms']:.1f}x |"
            )
        lines.append("")
    return "\n".join(lines)


def plot(rows: list[dict], k: int, path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: skipping the plot")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sorted({r["n"] for r in rows}):
        ivf = [r for r in rows if r["n"] == n and r["method"] == "ivf"]
        base = next(r for r in rows if r["n"] == n and r["method"] == "brute-force")
        line, = ax.plot([r["p50_ms"] for r in ivf], [r["recall"] for r in ivf], marker="o", label=f"IVF, N={n:,}")
        for r in ivf:
            ax.annotate(str(r["nprobe"]), (r["p50_ms"], r["recall"]), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, color=line.get_color())
        ax.scatter([base["p50_ms"]], [1.0], marker="*", s=140, color=line.get_color(), zorder=3)

    ax.scatter([], [], marker="*", s=140, color="grey", label="brute-force (exact)")
    ax.set_xscale("log")
    ax.set_xlabel("median query latency (ms, log scale)")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title("IVF recall/latency trade-off (point labels = nprobe)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="persisted VectorStore whose embeddings are benchmarked")
    parser.add_argument("--corpus", help="folder of .txt/.md files to embed into --store first (if it does not exist)")
    parser.add_argument("--limit", type=int, default=32_000, help="max chunks embedded from --corpus")
    parser.add_argument("--sizes", type=int, nargs="+", help="corpus sizes to benchmark (default: 5k, 10k, 20k, all)")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--description", help="where the corpus comes from, added to the report")
    args = parser.parse_args()

    if args.corpus and not os.path.isfile(os.path.join(args.store, "vectors.npy")):
        build_store(args.corpus, args.store, args.limit, args.seed)

    vectors, queries, source = load_store(args.store, args.queries, args.seed)
    if args.description:
        source = f"{args.description} ({source})"
    sizes = sorted({min(n, len(vectors)) for n in (args.sizes or [5_000, 10_000, 20_000, len(vectors)])})

    rows = []
    for n in sizes:
        print(f"N = {n:,} ...", flush=True)
        # Sous-échantillon aléatoire (fixé par la graine) pour les petites tailles
        subset = vectors[np.sort(np.random.default_rng(args.seed).permutation(len(vectors))[:n])]
        rows += run(subset, queries, args.k, args.nprobe, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    report = to_markdown(rows, args, source)
    with open(os.path.join(args.output_dir, "ivf.md"), "w", encoding="utf-8") as f:
        f.write(report)
    plot(rows, args.k, os.path.join(args.output_dir, "ivf_recall_latency.png"))
    print(report)


if __name__ == "__main__":
    main()
