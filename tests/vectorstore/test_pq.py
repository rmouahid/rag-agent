import numpy as np
import pytest

from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.ivf import normalize
from rag_agent.vectorstore.pq import INDEX_FILE, PQIndex, ProductQuantizer


def clustered_vectors(n=1000, dim=32, n_topics=12, seed=0):
    """Unit vectors grouped around a few random directions, like embeddings of distinct topics."""
    rng = np.random.default_rng(seed)
    topics = normalize(rng.normal(size=(n_topics, dim)))
    vectors = topics[rng.integers(n_topics, size=n)] + rng.normal(scale=0.15, size=(n, dim))
    return normalize(vectors).astype(np.float32)


def recall(index, vectors, queries, k=10, **search_args):
    found = 0
    for q in queries:
        exact, _ = cosine_top_k(vectors, q, k)
        approx, _ = index.search(q, k, **search_args)
        found += len(set(exact.tolist()) & set(approx.tolist()))
    return found / (k * len(queries))


# --- quantizer --------------------------------------------------------------


def test_codes_have_one_byte_per_subvector():
    vectors = clustered_vectors()
    pq = ProductQuantizer.train(vectors, m=8, n_bits=8)

    codes = pq.encode(vectors)

    assert codes.shape == (len(vectors), 8)
    assert codes.dtype == np.uint8
    assert pq.code_size == 8


def test_more_than_256_centroids_use_two_bytes():
    vectors = clustered_vectors(n=600, dim=8)
    pq = ProductQuantizer.train(vectors, m=2, n_bits=9, max_iter=3)

    assert pq.encode(vectors).dtype == np.uint16
    assert pq.code_size == 4


def test_each_subvector_is_coded_by_its_nearest_centroid():
    vectors = clustered_vectors()
    pq = ProductQuantizer.train(vectors, m=4, n_bits=4)

    codes = pq.encode(vectors)

    for j in range(pq.m):
        sub = vectors[:, j * pq.dsub : (j + 1) * pq.dsub].astype(np.float64)
        distances = ((sub[:, None, :] - pq.codebooks[j][None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_array_equal(codes[:, j], distances.argmin(axis=1))


def test_more_subvectors_reconstruct_more_accurately():
    vectors = clustered_vectors()

    errors = []
    for m in (2, 8, 32):
        pq = ProductQuantizer.train(vectors, m=m, n_bits=6)
        errors.append(float(((pq.decode(pq.encode(vectors)) - vectors) ** 2).sum(axis=1).mean()))

    assert errors[0] > errors[1] > errors[2]


def test_adc_score_is_the_dot_product_with_the_reconstruction():
    vectors = clustered_vectors()
    pq = ProductQuantizer.train(vectors, m=8, n_bits=6)
    codes = pq.encode(vectors)
    query = vectors[3].astype(np.float64)

    scores = pq.adc_scores(pq.inner_product_table(query), codes)

    np.testing.assert_allclose(scores, pq.decode(codes) @ query)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"m": 5}, "positive divisor"),
        ({"m": 0}, "positive divisor"),
        ({"n_bits": 0}, "n_bits"),
        ({"n_bits": 17}, "n_bits"),
        ({"n_bits": 11}, "training vectors"),
    ],
)
def test_invalid_training_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ProductQuantizer.train(clustered_vectors(), **kwargs)


def test_encoding_a_vector_of_the_wrong_dimension_fails():
    pq = ProductQuantizer.train(clustered_vectors(), m=8, n_bits=4)

    with pytest.raises(ValueError, match="dimension"):
        pq.encode(np.ones((2, 16)))


# --- index ------------------------------------------------------------------


def test_codes_take_far_less_memory_than_the_vectors():
    vectors = clustered_vectors(n=2000, dim=64)

    index = PQIndex.build(vectors, m=8, n_bits=8)

    assert index.codes.nbytes == 2000 * 8
    assert vectors.nbytes / index.codes.nbytes == 32  # 64 float32 = 256 bytes vs 8 bytes
    assert index.memory_bytes < vectors.nbytes


def test_search_finds_most_exact_neighbours():
    vectors = clustered_vectors()
    index = PQIndex.build(vectors, m=16, n_bits=8)

    assert recall(index, vectors, vectors[:50]) >= 0.6


def test_ivf_pq_probing_every_list_matches_pq_alone():
    vectors = clustered_vectors()
    flat = PQIndex.build(vectors, m=8, n_bits=6)
    ivf_pq = PQIndex.build(vectors, m=8, n_bits=6, n_lists=10)

    for q in vectors[:20]:
        ids_flat, scores_flat = flat.search(q, 10)
        ids_ivf, scores_ivf = ivf_pq.search(q, 10, nprobe=10)
        np.testing.assert_array_equal(ids_flat, ids_ivf)
        np.testing.assert_allclose(scores_flat, scores_ivf)


def test_ivf_pq_scores_only_the_probed_lists():
    vectors = clustered_vectors()
    index = PQIndex.build(vectors, m=8, n_bits=6, n_lists=10)
    query = vectors[0]

    ids, _ = index.search(query, 1000, nprobe=1)

    np.testing.assert_array_equal(np.sort(ids), np.sort(index.ivf.lists[index.ivf.probe(query, 1)[0]]))


def test_results_are_sorted_with_ties_broken_by_smallest_id():
    vectors = np.repeat(clustered_vectors(n=300, dim=16), 2, axis=0)  # every vector twice
    index = PQIndex.build(vectors, m=4, n_bits=6)

    ids, scores = index.search(vectors[10], 6)

    assert np.all(np.diff(scores) <= 0)
    for i in range(0, 6, 2):
        assert scores[i] == scores[i + 1] and ids[i] < ids[i + 1]


def test_k_larger_than_the_index_returns_everything():
    vectors = clustered_vectors(n=300, dim=16)
    index = PQIndex.build(vectors, m=4, n_bits=6)

    ids, _ = index.search(vectors[0], 1000)

    assert sorted(ids.tolist()) == list(range(300))


def test_added_vectors_are_encoded_and_searchable():
    vectors = clustered_vectors()
    index = PQIndex.build(vectors[:800], m=8, n_bits=6, n_lists=10)

    index.add(vectors[800:], np.arange(800, 1000))
    ids, _ = index.search(vectors[900], 1, nprobe=10)

    assert len(index) == 1000
    assert len(index.ivf) == 1000
    assert ids[0] == 900


def test_add_rejects_ids_that_do_not_extend_the_index():
    vectors = clustered_vectors()
    index = PQIndex.build(vectors[:800], m=8, n_bits=6)

    with pytest.raises(ValueError, match="extend the index"):
        index.add(vectors[800:810], np.arange(10))


def test_codebooks_are_trained_on_a_sample_when_capped():
    vectors = clustered_vectors(n=1000)

    index = PQIndex.build(vectors, m=8, n_bits=6, max_train=200)

    assert len(index) == 1000
    assert recall(index, vectors, vectors[:20]) > 0.3


@pytest.mark.parametrize("n_lists", [None, 10])
def test_save_and_load_round_trip(tmp_path, n_lists):
    vectors = clustered_vectors()
    index = PQIndex.build(vectors, m=8, n_bits=6, n_lists=n_lists)

    index.save(str(tmp_path))
    loaded = PQIndex.load(str(tmp_path))

    assert (tmp_path / INDEX_FILE).is_file()
    np.testing.assert_array_equal(loaded.codes, index.codes)
    assert (loaded.ivf is None) == (n_lists is None)
    for q in vectors[:5]:
        np.testing.assert_array_equal(loaded.search(q, 5)[0], index.search(q, 5)[0])
