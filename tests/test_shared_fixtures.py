import numpy as np
import pytest

from rag_agent.ingestion.md_loader import MarkdownLoader
from rag_agent.ingestion.pdf_loader import PDFLoader
from rag_agent.vectorstore.embedder import Embedder


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_offline_embeddings_keep_similarity_meaningful(offline_embeddings):
    embedder = Embedder()
    tower = embedder.embed("Who built the Eiffel Tower?")

    assert cosine(tower, embedder.embed("The Eiffel Tower was built by Gustave Eiffel.")) > 0.3
    assert cosine(tower, embedder.embed("Photosynthesis needs light")) == 0.0


def test_stub_llm_records_prompts_and_can_fail(stub_llm):
    assert stub_llm.generate("prompt") == "Gustave Eiffel's company."
    assert list(stub_llm.stream("again")) == ["Gustave", " Eiffel's", " company."]
    assert stub_llm.prompts == ["prompt", "again"]

    stub_llm.fail = True
    with pytest.raises(RuntimeError):
        stub_llm.generate("boom")


def test_sample_documents_load_with_the_real_loaders(sample_markdown, sample_pdf):
    markdown = MarkdownLoader().load(str(sample_markdown))
    pdf = PDFLoader().load(str(sample_pdf))

    assert "Gustave Eiffel" in markdown.text
    assert "Eiffel Tower is in Paris" in pdf.text
    assert "1889" in pdf.text
