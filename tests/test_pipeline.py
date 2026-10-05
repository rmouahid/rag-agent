import pytest

from rag_agent.chunking.chunker import FixedSizeChunker
from rag_agent.ingestion.document import Document
from rag_agent.pipeline import IngestionPipeline, IngestionResult, UnsupportedSourceError, source_type
from rag_agent.vectorstore.embedder import Embedder
from rag_agent.vectorstore.vector_store import VectorStore


@pytest.fixture
def pipeline(offline_embeddings):
    return IngestionPipeline(
        chunker=FixedSizeChunker(chunk_size=80, overlap=10),
        embedder=Embedder(),
        vector_store=VectorStore(index_type="brute-force"),
    )


def test_a_markdown_file_ends_up_indexed_and_searchable(pipeline, sample_markdown):
    result = pipeline.ingest(str(sample_markdown))

    assert isinstance(result, IngestionResult)
    assert result.source == str(sample_markdown)
    assert result.chunks_indexed == len(result.chunk_ids) > 1
    assert len(pipeline.vector_store) == result.chunks_indexed
    query = pipeline.embedder.embed("How tall is the tower in metres?")
    assert "330 metres" in pipeline.vector_store.search(query, top_k=1)[0]["text"]


def test_a_pdf_file_goes_through_the_same_pipeline(pipeline, sample_pdf):
    result = pipeline.ingest(str(sample_pdf))

    assert result.chunks_indexed >= 1
    assert {c["source"] for c in pipeline.vector_store.chunks} == {str(sample_pdf)}


def test_source_name_replaces_the_recorded_source(pipeline, sample_markdown):
    result = pipeline.ingest(str(sample_markdown), source_name="eiffel.md")

    assert result.source == "eiffel.md"
    assert {c["source"] for c in pipeline.vector_store.chunks} == {"eiffel.md"}


def test_reingesting_reports_duplicates_and_indexes_nothing(pipeline, sample_markdown):
    first = pipeline.ingest(str(sample_markdown))
    second = pipeline.ingest(str(sample_markdown))

    assert second.chunk_ids == []
    assert second.chunks_total == first.chunks_total
    assert second.duplicates == first.chunks_total
    assert len(pipeline.vector_store) == first.chunks_indexed


def test_an_already_loaded_document_can_be_ingested(pipeline, eiffel_text):
    result = pipeline.ingest_document(Document(text=eiffel_text, source="memory"))

    assert result.source == "memory"
    assert result.chunks_indexed >= 1


def test_an_empty_document_indexes_nothing(pipeline):
    result = pipeline.ingest_document(Document(text="", source="empty.md"))

    assert result.chunk_ids == []
    assert len(pipeline.vector_store) == 0


def test_unsupported_sources_are_refused_before_loading(pipeline, tmp_path):
    with pytest.raises(UnsupportedSourceError):
        pipeline.ingest(str(tmp_path / "notes.docx"))


@pytest.mark.parametrize(
    "source, kind",
    [("https://example.com", "web"), ("a.pdf", "pdf"), ("A.MD", "markdown"), ("b.markdown", "markdown")],
)
def test_source_type(source, kind):
    assert source_type(source) == kind


def test_the_agent_ingests_through_its_pipeline(vector_store_dir, stub_llm, sample_markdown):
    from rag_agent.generation.rag_agent import RAGAgent

    agent = RAGAgent(llm=stub_llm)
    ids = agent.ingest(str(sample_markdown))

    assert agent.pipeline.vector_store is agent.retriever.vector_store
    assert ids == [c["id"] for c in agent.retriever.vector_store.chunks]
