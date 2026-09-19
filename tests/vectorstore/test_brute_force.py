import math

import numpy as np
import pytest

from rag_agent.chunking.chunk import Chunk
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.vector_store import VectorStore


def naive_cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def test_ranks_by_cosine_with_exact_scores():
    matrix = np.array([[0, 1], [1, 0], [-1, 0], [1, 1]], dtype=np.float32)

    indices, scores = cosine_top_k(matrix, np.array([1, 0]), k=4)

    assert indices.tolist() == [1, 3, 0, 2]
    np.testing.assert_allclose(scores, [1.0, math.sqrt(0.5), 0.0, -1.0], atol=1e-12)


def test_matches_a_naive_reference_on_random_data():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 16)).astype(np.float32)
    query = rng.normal(size=16).astype(np.float32)

    indices, scores = cosine_top_k(matrix, query, k=10)

    expected = sorted(
        ((naive_cosine(row.tolist(), query.tolist()), i) for i, row in enumerate(matrix)),
        key=lambda pair: (-pair[0], pair[1]),
    )[:10]
    assert indices.tolist() == [i for _, i in expected]
    np.testing.assert_allclose(scores, [s for s, _ in expected], atol=1e-6)


def test_scores_are_sorted_descending():
    rng = np.random.default_rng(1)
    _, scores = cosine_top_k(rng.normal(size=(50, 8)), rng.normal(size=8), k=20)

    assert (np.diff(scores) <= 0).all()


def test_is_invariant_to_vector_magnitude():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(30, 8))
    query = rng.normal(size=8)

    base = cosine_top_k(matrix, query, k=5)
    scaled = cosine_top_k(matrix * rng.uniform(0.1, 100, size=(30, 1)), query * 7, k=5)

    assert base[0].tolist() == scaled[0].tolist()
    np.testing.assert_allclose(base[1], scaled[1], atol=1e-9)


def test_k_larger_than_corpus_returns_everything():
    matrix = np.eye(3)

    indices, scores = cosine_top_k(matrix, np.array([1.0, 0, 0]), k=50)

    assert len(indices) == len(scores) == 3


def test_ties_are_broken_by_lowest_index_even_across_the_cutoff():
    # rows 1, 3, 4 and 6 all score 1.0: the top-2 must be the two lowest indices
    matrix = np.array([[0, 1], [1, 0], [0, -1], [2, 0], [5, 0], [-1, 0], [1, 0]], dtype=np.float64)

    indices, _ = cosine_top_k(matrix, np.array([1.0, 0.0]), k=2)

    assert indices.tolist() == [1, 3]


def test_zero_vectors_score_zero_without_nan():
    matrix = np.array([[0, 0], [1, 0]], dtype=np.float32)

    indices, scores = cosine_top_k(matrix, np.array([1.0, 0.0]), k=2)
    assert indices.tolist() == [1, 0]
    assert scores.tolist() == [1.0, 0.0]

    indices, scores = cosine_top_k(matrix, np.zeros(2), k=2)
    assert indices.tolist() == [0, 1]
    assert scores.tolist() == [0.0, 0.0]


def test_rejects_invalid_arguments():
    matrix = np.ones((3, 4))

    with pytest.raises(ValueError, match="k must be"):
        cosine_top_k(matrix, np.ones(4), k=0)
    with pytest.raises(ValueError, match="dimension"):
        cosine_top_k(matrix, np.ones(5), k=1)
    with pytest.raises(ValueError, match="2-D"):
        cosine_top_k(np.ones(4), np.ones(4), k=1)


def test_vector_store_search_returns_chunks_with_scores():
    store = VectorStore()
    chunks = [Chunk(text=t, source="doc.md", position=i) for i, t in enumerate(["x", "y", "diag"])]
    store.add(chunks, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))

    results = store.search(np.array([1.0, 0.1]), top_k=2)

    assert [r["text"] for r in results] == ["x", "diag"]
    assert results[0]["source"] == "doc.md"
    assert results[0]["score"] > results[1]["score"]
    assert all(isinstance(r["score"], float) for r in results)


def test_vector_store_search_on_empty_store():
    assert VectorStore().search(np.ones(4)) == []
