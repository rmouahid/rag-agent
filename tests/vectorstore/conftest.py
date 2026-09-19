import numpy as np
import pytest

from rag_agent.vectorstore import embedder as embedder_module


class FakeModel:
    """Stand-in for SentenceTransformer: deterministic vectors, counts encoded texts."""

    DIM = 8

    def __init__(self, name):
        self.name = name
        self.encoded = []

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def encode(self, texts, convert_to_numpy=True, batch_size=32):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        self.encoded.extend(texts)
        vectors = np.stack(
            [np.random.default_rng(abs(hash(t)) % (2**32)).random(self.DIM, dtype=np.float32) for t in texts]
        )
        return vectors[0] if single else vectors


@pytest.fixture(autouse=True)
def fake_sentence_transformer(monkeypatch):
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeModel)
