"""
Benchmark de tous les backends de recherche : brute-force, IVF, HNSW, PQ
et IVF+PQ, sur les mêmes embeddings et les mêmes requêtes.

Pour chaque taille de corpus et chaque réglage, mesure :
- recall@k : part des k voisins exacts (brute-force) retrouvés
- latence  : temps médian et p95 d'une requête (une à la fois, comme dans
             le pipeline RAG)
- mémoire  : octets nécessaires pour répondre à une requête (vecteurs
             float32 et structure d'index ; PQ ne garde que ses codes)
- construction : temps de construction de l'index

Les embeddings viennent d'un VectorStore persisté (`--store`), construit au
besoin à partir d'un dossier de textes (`--corpus`), comme pour
ivf_benchmark.py, ou d'un jeu public au format .fvecs (`--fvecs`, par exemple
SIFT1M), normalisé L2 pour que la similarité cosinus donne les mêmes voisins
que la distance euclidienne.

Avec `--faiss` (groupe Poetry optionnel `bench`), FAISS IndexIVFFlat et
IndexHNSWFlat sont mesurés dans les mêmes conditions (un seul thread, mêmes
requêtes, mêmes réglages nprobe / ef) comme référence de l'état de l'art.

Usage :
    python benchmarks/ann_benchmark.py --store bench_store --sizes 32000
    python benchmarks/ann_benchmark.py --store big_store --sizes 10000 25000 50000 100000 --name scale
    python benchmarks/ann_benchmark.py --fvecs sift/sift_base.fvecs sift/sift_query.fvecs \
        --limit 100000 --faiss --name sift

Écrit docs/benchmarks/<name>.md (défaut : ann.md) et, si matplotlib est
installé, docs/benchmarks/<name>_recall_latency.png.
"""

import argparse
import os
import platform
import sys
from collections.abc import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from ivf_benchmark import build_store, load_store, timed  # noqa: E402

from rag_agent.vectorstore.brute_force import cosine_top_k  # noqa: E402
from rag_agent.vectorstore.hnsw import HNSWIndex  # noqa: E402
from rag_agent.vectorstore.ivf import IVFIndex  # noqa: E402
from rag_agent.vectorstore.pq import PQIndex  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "benchmarks")
METHODS = ("brute-force", "ivf", "hnsw", "pq", "ivf+pq", "faiss-ivf", "faiss-hnsw")
OWN_METHODS = METHODS[1:5]
WARMUP = 20  # requêtes non mesurées avant chaque réglage
PASSES = 2  # passes chronométrées par réglage, la meilleure est gardée


def read_fvecs(path: str, limit: int | None = None) -> np.ndarray:
    """Format .fvecs (TEXMEX) : pour chaque vecteur, sa dimension en int32 puis ses float32"""
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    count = -1 if limit is None else limit * (dim + 1)
    raw = np.fromfile(path, dtype=np.float32, count=count).reshape(-1, dim + 1)
    return np.ascontiguousarray(raw[:, 1:])


def normalized(vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def ivf_bytes(index: IVFIndex) -> int:
    return index.centroids.nbytes + sum(ids.nbytes for ids in index.lists)


def hnsw_bytes(index: HNSWIndex) -> int:
    """Graphe au format persisté : un id int64 par lien, toutes couches"""
    return 8 * sum(len(neighbours) for layer in index.links for neighbours in layer.values()) + index.levels.nbytes


def measure(
    search: Callable[[np.ndarray], np.ndarray], queries: np.ndarray, exact: list[set[int]]
) -> tuple[float, float, float]:
    """Rappel moyen, puis p50 et p95 en ms de la meilleure de PASSES passes.

    Sur une machine partagée, une passe isolée peut être ralentie par un autre
    processus : on garde la passe à la médiane la plus basse, qui reflète le
    coût de l'index plutôt que la charge du moment.
    """
    for q in queries[:WARMUP]:  # échauffement non mesuré : caches et allocations des premiers appels
        search(q)
    recalls, passes = [], []
    for attempt in range(PASSES):
        times = []
        for q, truth in zip(queries, exact):
            ids, elapsed = timed(search, q)
            if attempt == 0:
                recalls.append(len(truth & set(ids.tolist())) / len(truth))
            times.append(elapsed)
        passes.append(times)
    best = min(passes, key=np.median)
    return float(np.mean(recalls)), 1000 * float(np.median(best)), 1000 * float(np.percentile(best, 95))


def run(corpus: np.ndarray, queries: np.ndarray, args) -> list[dict]:
    corpus = corpus.astype(np.float32)
    n, k = len(corpus), args.k
    vectors_bytes = corpus.nbytes
    rows = []

    def row(method, setting, recall, p50, p95, memory, build_s):
        rows.append(
            {
                "n": n,
                "method": method,
                "setting": setting,
                "recall": recall,
                "p50_ms": p50,
                "p95_ms": p95,
                "memory": memory,
                "build_s": build_s,
            }
        )
        print(f"  {method:<11} {setting:<22} recall={recall:.3f} p50={p50:.2f} ms", flush=True)

    exact = [set(cosine_top_k(corpus, q, k)[0].tolist()) for q in queries]
    row(
        "brute-force", "exact scan", *measure(lambda q: cosine_top_k(corpus, q, k)[0], queries, exact), vectors_bytes, 0
    )

    if "ivf" in args.methods:
        index, build_s = timed(IVFIndex.build, corpus, None, args.seed)
        for nprobe in [p for p in args.nprobe if p <= index.n_lists]:
            stats = measure(lambda q, p=nprobe: index.search(corpus, q, k, p)[0], queries, exact)
            row("ivf", f"{index.n_lists} lists, nprobe={nprobe}", *stats, vectors_bytes + ivf_bytes(index), build_s)

    if "hnsw" in args.methods:
        index, build_s = timed(HNSWIndex.build, corpus)
        for ef in args.ef:
            stats = measure(lambda q, e=ef: index.search(corpus, q, k, e)[0], queries, exact)
            row("hnsw", f"M={index.m}, ef={ef}", *stats, vectors_bytes + hnsw_bytes(index), build_s)

    if "pq" in args.methods:
        for m in args.pq_m:
            index, build_s = timed(lambda m=m: PQIndex.build(corpus, m=m, seed=args.seed))
            stats = measure(lambda q, ix=index: ix.search(q, k)[0], queries, exact)
            row("pq", f"m={m} ({index.pq.code_size} B/vector)", *stats, index.memory_bytes, build_s)

    if "ivf+pq" in args.methods:
        n_lists = max(1, round(np.sqrt(n)))
        m = args.ivfpq_m
        index, build_s = timed(lambda: PQIndex.build(corpus, m=m, n_lists=n_lists, seed=args.seed))
        for nprobe in [p for p in args.nprobe if p <= n_lists]:
            stats = measure(lambda q, p=nprobe: index.search(q, k, p)[0], queries, exact)
            row("ivf+pq", f"m={m}, nprobe={nprobe}", *stats, index.memory_bytes, build_s)

    if args.faiss:
        run_faiss(corpus, queries, exact, args, row)

    return rows


def run_faiss(corpus: np.ndarray, queries: np.ndarray, exact: list[set[int]], args, row) -> None:
    """FAISS en référence : produit scalaire sur vecteurs normalisés, donc cosinus comme nos index"""
    import faiss

    faiss.omp_set_num_threads(1)  # comparaison à armes égales avec nos index mono-thread
    n, dim, k = len(corpus), corpus.shape[1], args.k
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    def search_with(index):
        return lambda q: index.search(q[None, :], k)[1][0]

    n_lists = max(1, round(np.sqrt(n)))  # même règle que IVFIndex
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_INNER_PRODUCT)
    _, build_s = timed(lambda: (ivf.train(corpus), ivf.add(corpus)))
    memory = len(faiss.serialize_index(ivf))
    for nprobe in [p for p in args.nprobe if p <= n_lists]:
        ivf.nprobe = nprobe
        row(
            "faiss-ivf",
            f"{n_lists} lists, nprobe={nprobe}",
            *measure(search_with(ivf), queries, exact),
            memory,
            build_s,
        )

    hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    _, build_s = timed(hnsw.add, corpus)
    memory = len(faiss.serialize_index(hnsw))
    for ef in args.ef:
        hnsw.hnsw.efSearch = ef
        row("faiss-hnsw", f"M=16, ef={ef}", *measure(search_with(hnsw), queries, exact), memory, build_s)


def operating_points(rows: list[dict], target: float) -> list[dict]:
    """Pour chaque taille et chaque méthode : le réglage le plus rapide qui atteint `target` de recall"""
    points = []
    for n in sorted({r["n"] for r in rows}):
        for method in METHODS:
            ok = [r for r in rows if r["n"] == n and r["method"] == method and r["recall"] >= target]
            if ok:
                points.append(min(ok, key=lambda r: r["p50_ms"]))
    return points


def mb(n_bytes: float) -> str:
    return f"{n_bytes / 2**20:.1f} MB" if n_bytes >= 2**20 else f"{n_bytes / 2**10:.0f} KB"


def to_markdown(rows: list[dict], args, source: str) -> str:
    k = args.k
    lines = [
        f"# Vector search backends benchmark{' at scale' if args.name == 'scale' else ''}",
        "",
        "_Generated by `benchmarks/ann_benchmark.py`; do not edit by hand._",
        "",
        f"- Data: {source}",
        f"- {args.queries} {'queries of the public query set' if args.fvecs else 'held-out queries'},"
        f" recall@{k} measured against the exact brute-force top-{k}",
        f"- Latency: one query at a time, median (p50) and 95th percentile (p95), after {WARMUP} warm-up"
        f" queries; best of {PASSES} timed passes",
        "- Memory: bytes needed to answer a query: float32 vectors plus the index structure"
        " (PQ and IVF+PQ keep only their codes, codebooks and lists)",
        f"- Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} CPU, "
        f"Python {platform.python_version()}, numpy {np.__version__}",
        "",
    ]

    points = operating_points(rows, args.target)
    if len({r["n"] for r in rows}) > 1:
        lines += [
            f"## Fastest setting reaching recall@{k} ≥ {args.target}",
            "",
            "| N | Method | Setting | Recall | p50 latency | Speed-up vs brute-force | Memory | Build |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for p in points:
            base = next(r for r in rows if r["n"] == p["n"] and r["method"] == "brute-force")
            lines.append(
                f"| {p['n']:,} | {p['method']} | {p['setting']} | {p['recall']:.3f} | {p['p50_ms']:.2f} ms "
                f"| {base['p50_ms'] / p['p50_ms']:.1f}x | {mb(p['memory'])} | {p['build_s']:.1f} s |"
            )
        lines.append("")

    for n in sorted({r["n"] for r in rows}):
        group = [r for r in rows if r["n"] == n]
        base = group[0]
        lines += [
            f"## N = {n:,} vectors",
            "",
            f"| Method | Setting | Recall@{k} | p50 latency | p95 latency | Speed-up (p50) | Memory | Build |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in group:
            lines.append(
                f"| {r['method']} | {r['setting']} | {r['recall']:.3f} | {r['p50_ms']:.2f} ms | {r['p95_ms']:.2f} ms "
                f"| {base['p50_ms'] / r['p50_ms']:.1f}x | {mb(r['memory'])} | {r['build_s']:.1f} s |"
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

    n = max(r["n"] for r in rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, marker in zip(METHODS, "*osD^xP"):
        points = sorted((r for r in rows if r["n"] == n and r["method"] == method), key=lambda r: r["p50_ms"])
        if points:
            ax.plot(
                [r["p50_ms"] for r in points],
                [r["recall"] for r in points],
                marker=marker,
                markersize=10 if marker == "*" else 5,
                linestyle="--" if method.startswith("faiss") else "-",
                label=method,
            )
    ax.set_xscale("log")
    ax.set_xlabel("median query latency (ms, log scale)")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title(f"Recall/latency trade-off of each backend (N = {n:,})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--store", help="persisted VectorStore whose embeddings are benchmarked")
    data.add_argument("--fvecs", nargs=2, metavar=("BASE", "QUERIES"), help="public .fvecs dataset, e.g. SIFT1M")
    parser.add_argument("--corpus", help="folder of .txt/.md files to embed into --store first (if it does not exist)")
    parser.add_argument(
        "--limit", type=int, default=32_000, help="max chunks embedded from --corpus, or base vectors read from --fvecs"
    )
    parser.add_argument("--sizes", type=int, nargs="+", help="corpus sizes to benchmark (default: all vectors)")
    parser.add_argument("--methods", nargs="+", choices=OWN_METHODS, default=list(OWN_METHODS))
    parser.add_argument("--faiss", action="store_true", help="also measure FAISS IndexIVFFlat and IndexHNSWFlat")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--pq-m", type=int, nargs="+", default=[16, 48, 96], help="PQ subvectors (divide the dim)")
    parser.add_argument("--ivfpq-m", type=int, default=48, help="PQ subvectors of the IVF+PQ index")
    parser.add_argument("--target", type=float, default=0.9, help="recall of the summary's operating points")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="ann", help="report name: docs/benchmarks/<name>.md")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--description", help="where the corpus comes from, added to the report")
    args = parser.parse_args()

    if args.corpus and not os.path.isfile(os.path.join(args.store, "vectors.npy")):
        build_store(args.corpus, args.store, args.limit, args.seed)

    if args.fvecs:
        vectors = normalized(read_fvecs(args.fvecs[0], args.limit))
        queries = normalized(read_fvecs(args.fvecs[1], args.queries))
        source = (
            f"{len(vectors):,} base vectors of `{os.path.basename(args.fvecs[0])}` and the first {len(queries)}"
            f" queries of `{os.path.basename(args.fvecs[1])}` ({vectors.shape[1]} dimensions), L2-normalised"
        )
    else:
        vectors, queries, source = load_store(args.store, args.queries, args.seed)
    if args.description:
        source = f"{args.description} ({source})"
    sizes = sorted({min(n, len(vectors)) for n in (args.sizes or [len(vectors)])})

    rows = []
    for n in sizes:
        print(f"N = {n:,} ...", flush=True)
        subset = vectors[np.sort(np.random.default_rng(args.seed).permutation(len(vectors))[:n])]
        rows += run(subset, queries, args)

    os.makedirs(args.output_dir, exist_ok=True)
    report = to_markdown(rows, args, source)
    with open(os.path.join(args.output_dir, f"{args.name}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    plot(rows, args.k, os.path.join(args.output_dir, f"{args.name}_recall_latency.png"))
    print(report)


if __name__ == "__main__":
    main()
