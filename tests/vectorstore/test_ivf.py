import numpy as np
import pytest

from rag_agent.chunking.chunk import Chunk
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
