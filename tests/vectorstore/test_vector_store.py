import dataclasses
import json
import os

import numpy as np
import pytest

from rag_agent.chunking.chunk import Chunk
from rag_agent.ingestion.document import Document
from rag_agent.vectorstore import embedder as embedder_module
from rag_agent.vectorstore import retriever as retriever_module
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.vectorstore.vector_store import VectorStore

DIM = 4


def make_chunks(*texts, source="doc.md", metadata=None):
    return [Chunk(text=t, source=source, position=i * 100, metadata=dict(metadata or {})) for i, t in enumerate(texts)]


def make_vectors(n, offset=0):
    return np.arange(offset, offset + n * DIM, dtype=np.float32).reshape(n, DIM) + 1


def test_add_appends_chunks_and_vectors():
    store = VectorStore()
    store.add(make_chunks("a", "b"), make_vectors(2))
    store.add(make_chunks("c", source="other.md"), make_vectors(1, offset=100))

    assert len(store) == 3
    assert [c["text"] for c in store.chunks] == ["a", "b", "c"]
    assert store.vectors.shape == (3, DIM)


def test_empty_store():
    store = VectorStore()

    assert len(store) == 0
    assert store.search(np.ones(DIM)) == []


def test_add_skips_chunk_ids_already_present():
    store = VectorStore()
    chunks = make_chunks("a", "b")
    store.add(chunks, make_vectors(2))

    store.add(chunks + make_chunks("c", source="other.md"), make_vectors(3, offset=50))

    assert [c["text"] for c in store.chunks] == ["a", "b", "c"]
    assert store.vectors.shape == (3, DIM)
    # the kept vector for "c" is the one that was passed alongside it
    np.testing.assert_array_equal(store.vectors[2], make_vectors(3, offset=50)[2])


def test_add_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one-to-one"):
        VectorStore().add(make_chunks("a", "b"), make_vectors(1))


def test_add_rejects_dimension_change():
    store = VectorStore()
    store.add(make_chunks("a"), make_vectors(1))

    with pytest.raises(ValueError, match="dimension"):
        store.add(make_chunks("b"), np.ones((1, DIM + 1), dtype=np.float32))
    assert len(store) == 1


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add(make_chunks("héllo", "b", metadata={"type": "pdf", "pages": [1, 2]}), make_vectors(2))
    store.save(str(tmp_path))

    loaded = VectorStore()
    loaded.load(str(tmp_path))

    assert loaded.chunks == store.chunks
    np.testing.assert_array_equal(loaded.vectors, store.vectors)
    assert loaded.chunks[0]["metadata"] == {"type": "pdf", "pages": [1, 2]}


def test_constructor_loads_an_existing_store(tmp_path):
    first = VectorStore(path=str(tmp_path))
    first.add(make_chunks("a", "b"), make_vectors(2))

    restarted = VectorStore(path=str(tmp_path))

    assert [c["text"] for c in restarted.chunks] == ["a", "b"]
    np.testing.assert_array_equal(restarted.vectors, first.vectors)


def test_constructor_with_missing_store_starts_empty(tmp_path):
    store = VectorStore(path=str(tmp_path / "does-not-exist-yet"))

    assert len(store) == 0
    assert not VectorStore.exists(str(tmp_path / "does-not-exist-yet"))


def test_add_persists_and_appends_to_an_existing_store(tmp_path):
    VectorStore(path=str(tmp_path)).add(make_chunks("a", "b"), make_vectors(2))

    reopened = VectorStore(path=str(tmp_path))
    reopened.add(make_chunks("c", source="other.md"), make_vectors(1, offset=100))

    final = VectorStore(path=str(tmp_path))
    assert [c["text"] for c in final.chunks] == ["a", "b", "c"]
    assert final.vectors.shape == (3, DIM)
    np.testing.assert_array_equal(final.vectors[:2], make_vectors(2))


def test_save_without_a_path_raises():
    with pytest.raises(ValueError, match="No path"):
        VectorStore().save()


def test_load_rejects_inconsistent_store(tmp_path):
    store = VectorStore()
    store.add(make_chunks("a", "b"), make_vectors(2))
    store.save(str(tmp_path))
    np.save(tmp_path / "current" / "vectors.npy", make_vectors(1))

    with pytest.raises(ValueError, match="Corrupt"):
        VectorStore().load(str(tmp_path))


def test_save_leaves_no_temp_files(tmp_path):
    store = VectorStore(path=str(tmp_path))
    store.add(make_chunks("a"), make_vectors(1))

    store.add(make_chunks("b"), make_vectors(1, offset=100))

    current = tmp_path / "current"
    assert set(tmp_path.iterdir()) == {current, current.resolve()}
    assert sorted(p.name for p in current.iterdir()) == ["chunks.json", "vectors.npy"]


def test_loads_legacy_store_and_publishes_next_save(tmp_path):
    original = VectorStore(index_type="brute-force")
    original.add(make_chunks("a"), make_vectors(1))
    np.save(tmp_path / "vectors.npy", original.vectors)
    (tmp_path / "chunks.json").write_text(json.dumps(original.chunks), encoding="utf-8")

    reopened = VectorStore(path=str(tmp_path), index_type="brute-force")
    assert reopened.chunks == original.chunks
    np.testing.assert_array_equal(reopened.vectors, original.vectors)

    reopened.add(make_chunks("b"), make_vectors(1, offset=100))
    loaded = VectorStore(path=str(tmp_path), index_type="brute-force")
    assert loaded.chunks == reopened.chunks
    np.testing.assert_array_equal(loaded.vectors, reopened.vectors)


@pytest.mark.parametrize("failed_file", ["ivf_index.npz", "current"])
def test_failed_save_keeps_previous_generation(tmp_path, monkeypatch, failed_file):
    settings = {"index_type": "ivf", "n_lists": 1, "ivf_min_vectors": 0}
    store = VectorStore(path=str(tmp_path), **settings)
    store.add(make_chunks("a", "b"), make_vectors(2))
    old_files = set(tmp_path.iterdir())
    original_replace = os.replace

    def fail_replace(src, dst):
        if os.path.basename(dst) == failed_file:
            raise OSError("simulated interruption")
        original_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated interruption"):
            store.add(make_chunks("c"), make_vectors(1, offset=100))

    reopened = VectorStore(**settings)
    reopened.load(str(tmp_path))
    assert [chunk["text"] for chunk in reopened.chunks] == ["a", "b"]
    np.testing.assert_array_equal(reopened.vectors, make_vectors(2))
    assert reopened.index.covers(2)
    assert set(tmp_path.iterdir()) == old_files

    store.save()
    reopened.load(str(tmp_path))
    assert reopened.chunks == store.chunks
    np.testing.assert_array_equal(reopened.vectors, store.vectors)
    assert reopened.index.covers(3)


def test_search_still_ranks_by_cosine_after_reload(tmp_path):
    store = VectorStore(path=str(tmp_path))
    store.add(
        make_chunks("x-axis", "y-axis"),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
    )

    results = VectorStore(path=str(tmp_path)).search(np.array([0.9, 0.1, 0, 0]), top_k=2)

    assert [r["text"] for r in results] == ["x-axis", "y-axis"]


def test_retriever_reuses_persisted_embeddings_across_restarts(tmp_path, monkeypatch):
    cfg = dataclasses.replace(
        retriever_module.config,
        vector_store_path=str(tmp_path / "store"),
        embedding_cache_dir="",
    )
    monkeypatch.setattr(retriever_module, "config", cfg)
    # Embedder reads its own module's config: without this, the fake
    # vectors would be written to the real embedding cache.
    monkeypatch.setattr(embedder_module, "config", cfg)
    doc = Document(text="Paris is the capital of France. " * 30, source="france.md", metadata={"type": "markdown"})

    first_run = Retriever(chunk_size=200, overlap=20, chunking_strategy="fixed")
    ids = first_run.ingest(doc)
    indexed = len(first_run.vector_store)
    assert indexed > 0
    assert ids == [c["id"] for c in first_run.vector_store.chunks]

    second_run = Retriever(chunk_size=200, overlap=20, chunking_strategy="fixed")
    assert len(second_run.vector_store) == indexed
    assert second_run.embedder.model.encoded == []  # nothing recomputed on startup

    assert second_run.ingest(doc) == []  # re-ingesting the same source is a no-op
    assert second_run.embedder.model.encoded == []
    assert len(second_run.vector_store) == indexed
    assert second_run.retrieve("capital of France", top_k=1)[0]["source"] == "france.md"


def test_an_empty_store_is_ready():
    assert VectorStore().is_ready


def test_a_store_with_one_vector_per_chunk_is_ready():
    store = VectorStore()
    store.add(make_chunks("a", "b"), make_vectors(2))
    assert store.is_ready


def test_a_store_whose_vectors_and_chunks_disagree_is_not_ready():
    store = VectorStore()
    store.add(make_chunks("a", "b"), make_vectors(2))
    store.chunks.append({"id": "orphan", "text": "c", "source": "doc.md", "metadata": {}})
    assert not store.is_ready


def test_an_ivf_store_without_its_index_is_not_ready():
    store = VectorStore(index_type="ivf", ivf_min_vectors=0)
    store.add(make_chunks("a", "b", "c"), make_vectors(3))
    assert store.is_ready
    store.index = None
    assert not store.is_ready
