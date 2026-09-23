import numpy as np
import pytest

from rag_agent.chunking.chunk import Chunk
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.ivf import INDEX_FILE, IVFIndex, default_n_lists, normalize
from rag_agent.vectorstore.vector_store import VectorStore


def clustered_vectors(n=600, dim=16, n_topics=12, seed=0):
    """Unit vectors grouped around a few random directions, like embeddings of distinct topics."""
    rng = np.random.default_rng(seed)
    topics = normalize(rng.normal(size=(n_topics, dim)))
    vectors = topics[rng.integers(n_topics, size=n)] + rng.normal(scale=0.15, size=(n, dim))
    return normalize(vectors).astype(np.float32)


def make_chunks(n, prefix="c"):
    return [Chunk(text=f"{prefix}{i}", source="doc.md", position=i) for i in range(n)]


# --- construction -----------------------------------------------------------


def test_every_vector_lands_in_exactly_one_list():
    vectors = clustered_vectors()

    index = IVFIndex.build(vectors, n_lists=20)

    assert index.n_lists == 20
    assert len(index) == len(vectors)
    assert index.covers(len(vectors))


def test_each_vector_is_filed_under_its_nearest_centroid():
    vectors = clustered_vectors()

    index = IVFIndex.build(vectors, n_lists=20)

    labels = index.assign(vectors)
    for list_no, ids in enumerate(index.lists):
        assert (labels[ids] == list_no).all()
        assert (np.diff(ids) > 0).all()  # posting lists are sorted by id


def test_default_number_of_lists_is_about_sqrt_n():
    assert default_n_lists(10_000) == 100
    assert default_n_lists(1) == 1
    assert IVFIndex.build(clustered_vectors(n=400)).n_lists == 20


def test_n_lists_is_capped_by_the_number_of_vectors():
    index = IVFIndex.build(clustered_vectors(n=5), n_lists=50)

    assert index.n_lists == 5


def test_build_is_deterministic_for_a_given_seed():
    vectors = clustered_vectors()

    a, b = IVFIndex.build(vectors, n_lists=10, seed=3), IVFIndex.build(vectors, n_lists=10, seed=3)

    np.testing.assert_array_equal(a.centroids, b.centroids)
    assert all(np.array_equal(x, y) for x, y in zip(a.lists, b.lists))


def test_clustering_is_by_direction_not_magnitude():
    vectors = clustered_vectors(n=200)
    scaled = vectors * np.random.default_rng(1).uniform(0.1, 50, size=(200, 1))

    a, b = IVFIndex.build(vectors, n_lists=8), IVFIndex.build(scaled, n_lists=8)

    assert all(np.array_equal(x, y) for x, y in zip(a.lists, b.lists))


def test_build_rejects_an_empty_matrix():
    with pytest.raises(ValueError, match="cannot build"):
        IVFIndex.build(np.empty((0, 4)))


# --- persistence --------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    index = IVFIndex.build(clustered_vectors(), n_lists=15)
    index.save(str(tmp_path))

    loaded = IVFIndex.load(str(tmp_path))

    np.testing.assert_array_equal(loaded.centroids, index.centroids)
    assert all(np.array_equal(x, y) for x, y in zip(loaded.lists, index.lists))
    assert sorted(p.name for p in tmp_path.iterdir()) == [INDEX_FILE]


def test_covers_detects_an_index_out_of_sync_with_the_store():
    index = IVFIndex.build(clustered_vectors(n=50), n_lists=5)

    assert index.covers(50)
    assert not index.covers(51)
    index.lists[0] = index.lists[0][:-1]
    assert not index.covers(50)


# --- VectorStore integration ------------------------------------------------------


def test_vector_store_builds_and_persists_the_index_alongside_the_store(tmp_path):
    vectors = clustered_vectors(n=100)
    store = VectorStore(path=str(tmp_path), index_type="ivf", n_lists=10)
    store.add(make_chunks(100), vectors)

    assert store.index.n_lists == 10
    assert store.index.covers(100)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chunks.json", INDEX_FILE, "vectors.npy"]


def test_vector_store_reloads_the_persisted_index_without_rebuilding(tmp_path, monkeypatch):
    store = VectorStore(path=str(tmp_path), index_type="ivf", n_lists=10)
    store.add(make_chunks(100), clustered_vectors(n=100))

    def fail(*args, **kwargs):
        raise AssertionError("the index should be loaded, not rebuilt")

    monkeypatch.setattr(IVFIndex, "build", fail)
    reopened = VectorStore(path=str(tmp_path), index_type="ivf")

    np.testing.assert_array_equal(reopened.index.centroids, store.index.centroids)


def test_vector_store_rebuilds_a_stale_or_missing_index_on_load(tmp_path):
    VectorStore(path=str(tmp_path), index_type="brute-force").add(make_chunks(60), clustered_vectors(n=60))
    assert not IVFIndex.exists(str(tmp_path))

    reopened = VectorStore(path=str(tmp_path), index_type="ivf", n_lists=6)

    assert reopened.index.covers(60)
    assert IVFIndex.exists(str(tmp_path))


def test_vector_store_rebuilds_a_corrupt_index_on_load(tmp_path):
    VectorStore(path=str(tmp_path), index_type="ivf", n_lists=6).add(make_chunks(60), clustered_vectors(n=60))
    (tmp_path / INDEX_FILE).write_bytes(b"not an npz file")

    reopened = VectorStore(path=str(tmp_path), index_type="ivf", n_lists=6)

    assert reopened.index.covers(60)


def test_brute_force_store_has_no_index():
    store = VectorStore(index_type="brute-force")
    store.add(make_chunks(10), clustered_vectors(n=10))

    assert store.index is None


def test_vector_store_rejects_an_unknown_index_type():
    with pytest.raises(ValueError, match="index_type"):
        VectorStore(index_type="hnsw")


# --- search -------------------------------------------------------------------------


def recall(approx, exact):
    return len(set(approx.tolist()) & set(exact.tolist())) / len(exact)


def test_probe_returns_the_nearest_lists_first():
    index = IVFIndex.build(clustered_vectors(), n_lists=20)
    query = clustered_vectors(n=1, seed=9)[0]

    probed = index.probe(query, nprobe=5)

    distances = ((index.centroids - normalize(query)) ** 2).sum(axis=1)
    np.testing.assert_array_equal(probed, np.argsort(distances, kind="stable")[:5])


def test_probing_every_list_is_exactly_the_brute_force_result():
    vectors = clustered_vectors()
    index = IVFIndex.build(vectors, n_lists=20)
    rng = np.random.default_rng(1)

    for query in rng.normal(size=(20, vectors.shape[1])):
        ids, scores = index.search(vectors, query, k=10, nprobe=20)
        exact_ids, exact_scores = cosine_top_k(vectors, query, 10)
        np.testing.assert_array_equal(ids, exact_ids)
        np.testing.assert_array_equal(scores, exact_scores)


def test_results_only_come_from_the_probed_lists():
    vectors = clustered_vectors()
    index = IVFIndex.build(vectors, n_lists=20)
    query = vectors[0]

    ids, scores = index.search(vectors, query, k=10, nprobe=2)

    allowed = set(np.concatenate([index.lists[i] for i in index.probe(query, 2)]).tolist())
    assert set(ids.tolist()) <= allowed
    assert (np.diff(scores) <= 0).all()


def test_recall_grows_with_nprobe_and_is_high_on_clustered_data():
    vectors = clustered_vectors(n=2000, n_topics=40)
    index = IVFIndex.build(vectors, n_lists=40)
    queries = clustered_vectors(n=50, n_topics=40, seed=7)

    def mean_recall(nprobe):
        return np.mean([
            recall(index.search(vectors, q, 10, nprobe)[0], cosine_top_k(vectors, q, 10)[0]) for q in queries
        ])

    recalls = [mean_recall(p) for p in (1, 4, 16, 40)]
    assert recalls == sorted(recalls)
    assert recalls[-1] == 1.0
    assert recalls[1] >= 0.8
    assert recalls[2] >= 0.95


def test_search_rejects_invalid_arguments():
    vectors = clustered_vectors(n=50)
    index = IVFIndex.build(vectors, n_lists=5)

    with pytest.raises(ValueError, match="nprobe"):
        index.search(vectors, vectors[0], k=5, nprobe=0)
    with pytest.raises(ValueError, match="dimension"):
        index.search(vectors, np.ones(3), k=5, nprobe=1)


def test_vector_store_search_goes_through_the_ivf_index(monkeypatch):
    vectors = clustered_vectors(n=200)
    store = VectorStore(index_type="ivf", n_lists=10, nprobe=3)
    store.add(make_chunks(200), vectors)
    calls = []
    original = IVFIndex.search

    def spy(self, matrix, query, k, nprobe):
        calls.append(nprobe)
        return original(self, matrix, query, k, nprobe)

    monkeypatch.setattr(IVFIndex, "search", spy)
    results = store.search(vectors[5], top_k=3)
    store.search(vectors[5], top_k=3, nprobe=10)

    assert calls == [3, 10]
    assert results[0]["text"] == "c5"
    assert results[0]["score"] == pytest.approx(1.0)


def test_vector_store_nprobe_defaults_to_config():
    from rag_agent.config import config

    assert VectorStore(index_type="ivf").nprobe == config.nprobe


# --- incremental updates -------------------------------------------------------------


def test_add_files_new_vectors_under_their_nearest_existing_centroid():
    vectors = clustered_vectors(n=300)
    index = IVFIndex.build(vectors[:200], n_lists=10)
    centroids = index.centroids.copy()

    index.add(vectors[200:], np.arange(200, 300))

    np.testing.assert_array_equal(index.centroids, centroids)  # no re-clustering
    assert index.covers(300)
    labels = index.assign(vectors)
    for list_no, ids in enumerate(index.lists):
        assert (labels[ids] == list_no).all()
        assert (np.diff(ids) > 0).all()


def test_drift_is_the_share_of_incrementally_added_vectors():
    vectors = clustered_vectors(n=300)
    index = IVFIndex.build(vectors[:200], n_lists=10)
    assert index.drift == 0.0

    index.add(vectors[200:250], np.arange(200, 250))
    assert index.drift == pytest.approx(50 / 250)


def test_add_rejects_mismatched_ids():
    index = IVFIndex.build(clustered_vectors(n=20), n_lists=2)

    with pytest.raises(ValueError, match="one-to-one"):
        index.add(clustered_vectors(n=3), np.arange(20, 22))


def test_incremental_index_still_finds_the_added_vectors():
    vectors = clustered_vectors(n=400)
    index = IVFIndex.build(vectors[:300], n_lists=12)
    index.add(vectors[300:], np.arange(300, 400))

    for i in range(300, 400, 10):
        ids, _ = index.search(vectors, vectors[i], k=1, nprobe=1)
        assert ids[0] == i


def test_n_added_survives_save_and_load(tmp_path):
    vectors = clustered_vectors(n=120)
    index = IVFIndex.build(vectors[:100], n_lists=5)
    index.add(vectors[100:], np.arange(100, 120))
    index.save(str(tmp_path))

    assert IVFIndex.load(str(tmp_path)).n_added == 20


def test_vector_store_adds_incrementally_below_the_rebuild_threshold(monkeypatch):
    vectors = clustered_vectors(n=110)
    store = VectorStore(index_type="ivf", n_lists=8, rebuild_threshold=0.2)
    store.add(make_chunks(100), vectors[:100])
    built = store.index

    monkeypatch.setattr(IVFIndex, "build", lambda *a, **k: pytest.fail("should not rebuild"))
    store.add(make_chunks(10, prefix="new"), vectors[100:])

    assert store.index is built
    assert store.index.covers(110)
    assert store.search(vectors[105], top_k=1)[0]["text"] == "new5"


def test_vector_store_rebuilds_once_drift_exceeds_the_threshold():
    vectors = clustered_vectors(n=200)
    store = VectorStore(index_type="ivf", n_lists=8, rebuild_threshold=0.2)
    store.add(make_chunks(100), vectors[:100])

    store.add(make_chunks(20, prefix="a"), vectors[100:120])  # drift 20/120 <= 0.2: incremental
    assert store.index.n_added == 20

    store.add(make_chunks(10, prefix="b"), vectors[120:130])  # drift 30/130 > 0.2: rebuild
    assert store.index.n_added == 0
    assert store.index.covers(130)


def test_vector_store_rebuild_index_resets_drift():
    vectors = clustered_vectors(n=150)
    store = VectorStore(index_type="ivf", rebuild_threshold=1.0)
    store.add(make_chunks(100), vectors[:100])
    store.add(make_chunks(50, prefix="x"), vectors[100:])
    assert store.index.drift > 0

    store.rebuild_index()

    assert store.index.drift == 0.0
    assert store.index.n_lists == 12  # auto n_lists recomputed on the grown corpus (~sqrt(150))
