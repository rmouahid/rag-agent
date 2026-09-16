import pytest

from rag_agent.chunking.chunk import Chunk
from rag_agent.chunking.chunker import FixedSizeChunker
from rag_agent.ingestion.document import Document


def test_no_chunk_exceeds_max_size():
    doc = Document(text="x" * 1000, source="t.txt")
    chunks = FixedSizeChunker(chunk_size=100, overlap=20).chunk(doc)

    assert all(len(c.text) <= 100 for c in chunks)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_overlap_between_consecutive_chunks():
    doc = Document(text="abcdefghij" * 10, source="t.txt")
    chunker = FixedSizeChunker(chunk_size=30, overlap=10)
    chunks = chunker.chunk(doc)

    assert chunks[1].position == chunks[0].position + (30 - 10)


def test_empty_document_yields_no_chunks():
    doc = Document(text="", source="t.txt")
    assert FixedSizeChunker(chunk_size=50, overlap=10).chunk(doc) == []


def test_whitespace_only_document_yields_no_chunks():
    doc = Document(text="   \n\n   ", source="t.txt")
    assert FixedSizeChunker(chunk_size=50, overlap=10).chunk(doc) == []


def test_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, overlap=50)


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0, overlap=0)


def test_chunk_carries_source_and_metadata():
    doc = Document(text="hello world", source="t.txt", metadata={"type": "text"})
    chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk(doc)

    assert chunks[0].source == "t.txt"
    assert chunks[0].metadata == {"type": "text"}
