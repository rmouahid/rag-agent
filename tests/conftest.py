"""
Fixtures shared by the whole suite. They let the real pipeline run
anywhere, CI included, by replacing the two components that need external
resources:
- the LLM, which would need the GGUF weights (`stub_llm`);
- the sentence-transformers model, which would be downloaded
  (`offline_embeddings`). A bag-of-words model keeps similarity
  meaningful: texts sharing words are close, texts sharing none are
  orthogonal.
Plus sample documents to ingest (`eiffel_text`, `sample_markdown`,
`sample_pdf`, `make_pdf`).
"""

import dataclasses
import re
import zlib

import numpy as np
import pymupdf
import pytest

from rag_agent.vectorstore import embedder as embedder_module
from rag_agent.vectorstore import retriever as retriever_module

EIFFEL = (
    "# Eiffel Tower\n\n"
    "The Eiffel Tower was built by Gustave Eiffel's company for the 1889 World's Fair in Paris. "
    "The tower is 330 metres tall and was the tallest structure in the world until 1930.\n"
)


class BagOfWordsModel:
    """Stand-in for SentenceTransformer: one dimension per hashed word."""

    DIM = 256

    def __init__(self, name):
        self.name = name

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def _vector(self, text):
        vector = np.zeros(self.DIM, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.DIM] += 1.0
        return vector

    def encode(self, texts, convert_to_numpy=True, batch_size=32):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.stack([self._vector(t) for t in texts])


class StubLLM:
    """Stand-in for LocalLLM: fixed answer, records every prompt, can be made to fail."""

    model_path = "models/stub.gguf"
    is_loaded = True
    prompt_budget = 3000

    def __init__(self, answer=("Gustave", " Eiffel's", " company.")):
        self.answer = list(answer)
        self.prompts = []
        self.fail = False

    def count_tokens(self, text):
        return len(text.split())

    def _answer(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("llama_decode returned -3")
        return self.answer

    def generate(self, prompt):
        return "".join(self._answer(prompt))

    def stream(self, prompt):
        yield from self._answer(prompt)


@pytest.fixture
def stub_llm():
    return StubLLM()


@pytest.fixture
def offline_embeddings(monkeypatch):
    """
    Bag-of-words embeddings instead of the sentence-transformers model, and
    no on-disk embedding cache
    """
    monkeypatch.setattr(embedder_module, "SentenceTransformer", BagOfWordsModel)
    monkeypatch.setattr(
        embedder_module, "config", dataclasses.replace(embedder_module.config, embedding_cache_dir="")
    )


@pytest.fixture
def vector_store_dir(tmp_path, monkeypatch, offline_embeddings):
    """
    Empty directory used as VECTOR_STORE_PATH by every Retriever (and so
    every RAGAgent) built during the test
    """
    path = tmp_path / "vector_db"
    monkeypatch.setattr(
        retriever_module, "config", dataclasses.replace(retriever_module.config, vector_store_path=str(path))
    )
    return path


@pytest.fixture
def eiffel_text():
    return EIFFEL


@pytest.fixture
def make_pdf():
    """Builds an in-memory PDF with one page per text."""

    def make(*pages: str) -> bytes:
        doc = pymupdf.open()
        for text in pages:
            doc.new_page().insert_text((72, 72), text)
        return doc.tobytes()

    return make


@pytest.fixture
def sample_markdown(tmp_path):
    path = tmp_path / "eiffel.md"
    path.write_text(EIFFEL, encoding="utf-8")
    return path


@pytest.fixture
def sample_pdf(tmp_path, make_pdf):
    path = tmp_path / "tower.pdf"
    path.write_bytes(make_pdf("The Eiffel Tower is in Paris.", "It was completed in 1889."))
    return path
