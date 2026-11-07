import numpy as np
import pytest

from rag_agent.chunking.chunk import Chunk
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.hnsw import INDEX_FILE, HNSWIndex
from rag_agent.vectorstore.ivf import normalize
from rag_agent.vectorstore.vector_store import VectorStore


def clustered_vectors(n=1000, dim=32, n_topics=20, seed=0):
    """Unit vectors grouped around random directions, like embeddings of distinct topics."""
    rng = np.random.default_rng(seed)
    topics = normalize(rng.normal(size=(n_topics, dim)))
    vectors = topics[rng.integers(n_topics, size=n)] + rng.normal(scale=0.25, size=(n, dim))
    return normalize(vectors).astype(np.float32)


def recall_at_k(index, vectors, queries, k=10, ef=None):
    hits = 0
    for q in queries:
        exact, _ = cosine_top_k(vectors, q, k)
        found, _ = index.search(vectors, q, k, ef or index.ef_search)
        hits += len(set(exact.tolist()) & set(found.tolist()))
    return hits / (k * len(queries))


def make_chunks(n, prefix="c"):
    return [Chunk(text=f"{prefix}{i}", source="doc.md", position=i) for i in range(n)]


@pytest.fixture(scope="module")
def vectors():
    return clustered_vectors()


@pytest.fixture(scope="module")
def queries():
    return clustered_vectors(n=50, seed=1)


@pytest.fixture(scope="module")
def index(vectors):
    return HNSWIndex.build(vectors, m=12, ef_construction=100, ef_search=64)


# --- construction -----------------------------------------------------------


def test_every_vector_is_in_the_graph(index, vectors):
    assert len(index) == len(vectors)
    assert index.covers(len(vectors))


def test_degrees_are_bounded(index):
    for level, links in enumerate(index.links):
        cap = index.max_links(level)
        assert all(len(neighbours) <= cap for neighbours in links.values())


def test_upper_layers_get_sparser(index):
    sizes = [len(links) for links in index.links]
    assert sizes[0] == len(index)
    assert all(a >= b for a, b in zip(sizes, sizes[1:]))
    assert index.entry_point in index.links[-1]


def test_building_is_deterministic_for_a_seed(vectors):
    a = HNSWIndex.build(vectors[:200], m=8, ef_construction=50, seed=3)
    b = HNSWIndex.build(vectors[:200], m=8, ef_construction=50, seed=3)

    assert a.levels.tolist() == b.levels.tolist()
    assert a.links == b.links


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError, match="m"):
        HNSWIndex(dim=4, m=1)
    with pytest.raises(ValueError, match="ef_construction"):
        HNSWIndex(dim=4, ef_construction=0)


# --- search -----------------------------------------------------------------


def test_recall_is_high_on_clustered_data(index, vectors, queries):
    assert recall_at_k(index, vectors, queries) >= 0.95


def test_a_larger_ef_does_not_lower_recall(index, vectors, queries):
    assert recall_at_k(index, vectors, queries, ef=200) >= recall_at_k(index, vectors, queries, ef=10)


def test_an_indexed_vector_finds_itself_first(index, vectors):
    for i in (0, 123, 999):
        ids, scores = index.search(vectors, vectors[i], 1, 64)
        assert ids[0] == i
        assert scores[0] == pytest.approx(1.0)


def test_results_are_sorted_like_brute_force(index, vectors, queries):
    ids, scores = index.search(vectors, queries[0], 10, 64)

    assert np.all(np.diff(scores) <= 0)
    exact_scores = vectors[ids].astype(np.float64) @ (queries[0] / np.linalg.norm(queries[0]))
    np.testing.assert_allclose(scores, exact_scores, rtol=1e-6)


def test_k_larger_than_the_index_returns_everything(vectors):
    small = HNSWIndex.build(vectors[:5], m=4, ef_construction=10)

    ids, _ = small.search(vectors[:5], vectors[0], 10, 10)

    assert sorted(ids.tolist()) == [0, 1, 2, 3, 4]


def test_an_empty_index_returns_nothing():
    ids, scores = HNSWIndex(dim=4).search(np.empty((0, 4)), np.ones(4), 3, 10)

    assert len(ids) == 0 and len(scores) == 0


def test_query_dimension_is_checked(index, vectors):
    with pytest.raises(ValueError, match="dimension"):
        index.search(vectors, np.ones(3), 5, 10)


# --- incremental insertion ------------------------------------------------------


def test_inserting_into_an_existing_graph_keeps_recall(vectors, queries):
    grown = HNSWIndex.build(vectors[:500], m=12, ef_construction=100, ef_search=64)
    grown.add(vectors[500:], np.arange(500, len(vectors)))

    assert grown.covers(len(vectors))
    assert recall_at_k(grown, vectors, queries) >= 0.95


def test_ids_must_extend_the_index(vectors):
    index = HNSWIndex.build(vectors[:10], m=4, ef_construction=10)
    with pytest.raises(ValueError, match="ids"):
        index.add(vectors[10:12], np.array([3, 4]))


# --- persistence ------------------------------------------------------------------


def test_save_and_load_give_the_same_results(index, vectors, queries, tmp_path):
    index.save(str(tmp_path))
    loaded = HNSWIndex.load(str(tmp_path), vectors)

    assert HNSWIndex.exists(str(tmp_path))
    assert loaded.links == index.links
    for q in queries[:5]:
        np.testing.assert_array_equal(loaded.search(vectors, q, 10, 64)[0], index.search(vectors, q, 10, 64)[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == [INDEX_FILE]


def test_a_corrupt_file_is_refused(tmp_path, vectors):
    (tmp_path / INDEX_FILE).write_bytes(b"not an npz file")
    with pytest.raises((ValueError, OSError)):
        HNSWIndex.load(str(tmp_path), vectors)


# --- VectorStore integration ------------------------------------------------------


def test_vector_store_searches_through_the_hnsw_graph(vectors, monkeypatch):
    store = VectorStore(index_type="hnsw", ivf_min_vectors=0)
    store.add(make_chunks(300), vectors[:300])
    calls = []
    search = store.index.search
    monkeypatch.setattr(store.index, "search", lambda *a, **k: calls.append(a) or search(*a, **k))

    results = store.search(vectors[42], top_k=3)

    assert calls
    assert results[0]["id"] == store.chunks[42]["id"]


def test_vector_store_grows_the_graph_incrementally(vectors, monkeypatch):
    store = VectorStore(index_type="hnsw", ivf_min_vectors=0)
    store.add(make_chunks(100), vectors[:100])
    first = store.index

    def fail(*args, **kwargs):
        raise AssertionError("the graph should grow, not be rebuilt")

    monkeypatch.setattr(HNSWIndex, "build", fail)
    store.add(make_chunks(100, prefix="d"), vectors[100:200])

    assert store.index is first
    assert store.index.covers(200)


def test_vector_store_reloads_the_graph_without_rebuilding(vectors, tmp_path, monkeypatch):
    store = VectorStore(path=str(tmp_path), index_type="hnsw", ivf_min_vectors=0)
    store.add(make_chunks(150), vectors[:150])

    def fail(*args, **kwargs):
        raise AssertionError("the graph should be loaded, not rebuilt")

    monkeypatch.setattr(HNSWIndex, "build", fail)
    reopened = VectorStore(path=str(tmp_path), index_type="hnsw", ivf_min_vectors=0)

    assert reopened.index.links == store.index.links
    assert reopened.is_ready


def test_a_stale_graph_is_rebuilt_on_load(vectors, tmp_path):
    VectorStore(path=str(tmp_path), index_type="brute-force").add(make_chunks(80), vectors[:80])

    reopened = VectorStore(path=str(tmp_path), index_type="hnsw", ivf_min_vectors=0)

    assert reopened.index.covers(80)
    assert HNSWIndex.exists(str(tmp_path))


def test_hnsw_parameters_come_from_the_configuration(vectors, monkeypatch):
    import dataclasses

    from rag_agent.vectorstore import vector_store as vector_store_module

    config = dataclasses.replace(
        vector_store_module.config,
        index_type="hnsw",
        ivf_min_vectors=0,
        hnsw_m=6,
        hnsw_ef_construction=40,
        hnsw_ef_search=20,
    )
    monkeypatch.setattr(vector_store_module, "config", config)
    store = VectorStore()
    store.add(make_chunks(50), vectors[:50])

    assert (store.index.m, store.index.ef_construction, store.index.ef_search) == (6, 40, 20)
