import dataclasses

import numpy as np

from rag_agent.chunking.chunk import Chunk
from rag_agent.vectorstore import embedder as embedder_module
from rag_agent.vectorstore.embedder import Embedder, EmbeddingCache

DIM = 8  # FakeModel.DIM, see conftest.py


def make_chunks(*texts, source="doc.md"):
    return [Chunk(text=t, source=source, position=i * 100) for i, t in enumerate(texts)]


def test_embed_chunks_returns_one_row_per_chunk_in_order(tmp_path):
    emb = Embedder(cache_dir=str(tmp_path))
    chunks = make_chunks("alpha", "beta", "gamma")

    result = emb.embed_chunks(chunks)

    assert result.shape == (3, DIM)
    for row, chunk in zip(result, chunks):
        np.testing.assert_array_equal(row, emb.embed(chunk.text))


def test_embed_chunks_empty_batch(tmp_path):
    emb = Embedder(cache_dir=str(tmp_path))

    assert emb.embed_chunks([]).shape == (0, DIM)
    assert emb.model.encoded == []


def test_cached_chunks_are_not_re_encoded(tmp_path):
    emb = Embedder(cache_dir=str(tmp_path))
    chunks = make_chunks("alpha", "beta")

    first = emb.embed_chunks(chunks)
    encoded_after_first = list(emb.model.encoded)
    second = emb.embed_chunks(chunks)

    assert emb.model.encoded == encoded_after_first
    np.testing.assert_array_equal(first, second)


def test_only_new_chunks_are_encoded(tmp_path):
    emb = Embedder(cache_dir=str(tmp_path))
    emb.embed_chunks(make_chunks("alpha", "beta"))
    emb.model.encoded.clear()

    emb.embed_chunks(make_chunks("alpha", "beta", "gamma"))

    assert emb.model.encoded == ["gamma"]


def test_duplicate_ids_in_a_batch_are_encoded_once(tmp_path):
    emb = Embedder(cache_dir=str(tmp_path))
    chunk = Chunk(text="alpha", source="doc.md", position=0)

    result = emb.embed_chunks([chunk, chunk])

    assert emb.model.encoded == ["alpha"]
    np.testing.assert_array_equal(result[0], result[1])


def test_cache_persists_across_runs(tmp_path):
    chunks = make_chunks("alpha", "beta")
    first_run = Embedder(cache_dir=str(tmp_path))
    expected = first_run.embed_chunks(chunks)

    second_run = Embedder(cache_dir=str(tmp_path))
    result = second_run.embed_chunks(chunks)

    assert second_run.model.encoded == []
    np.testing.assert_array_equal(result, expected)


def test_cache_is_separate_per_model(tmp_path):
    Embedder(model_name="model-a", cache_dir=str(tmp_path)).embed_chunks(make_chunks("alpha"))

    other = Embedder(model_name="org/model-b", cache_dir=str(tmp_path))
    other.embed_chunks(make_chunks("alpha"))

    assert other.model.encoded == ["alpha"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model-a.npz", "org_model-b.npz"]


def test_empty_cache_dir_disables_persistence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    emb = Embedder(cache_dir="")

    emb.embed_chunks(make_chunks("alpha"))
    emb.embed_chunks(make_chunks("alpha"))

    assert emb.model.encoded == ["alpha"]  # in-memory cache still works
    assert list(tmp_path.iterdir()) == []


def test_model_name_defaults_to_config(tmp_path, monkeypatch):
    patched = dataclasses.replace(embedder_module.config, embedding_model="configured-model")
    monkeypatch.setattr(embedder_module, "config", patched)

    emb = Embedder(cache_dir=str(tmp_path))

    assert emb.model_name == "configured-model"
    assert emb.model.name == "configured-model"


def test_corrupt_cache_file_is_rebuilt(tmp_path):
    (tmp_path / "all-MiniLM-L6-v2.npz").write_bytes(b"not a real npz")

    emb = Embedder(cache_dir=str(tmp_path))
    result = emb.embed_chunks(make_chunks("alpha"))

    assert result.shape == (1, DIM)
    assert emb.model.encoded == ["alpha"]


def test_embedding_cache_roundtrip(tmp_path):
    path = str(tmp_path / "sub" / "cache.npz")
    cache = EmbeddingCache(path)
    cache.put("abc", np.arange(4, dtype=np.float32))
    cache.save()

    reloaded = EmbeddingCache(path)

    assert "abc" in reloaded
    assert len(reloaded) == 1
    np.testing.assert_array_equal(reloaded.get("abc"), np.arange(4, dtype=np.float32))
