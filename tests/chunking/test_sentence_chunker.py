import pytest

from rag_agent.chunking.chunk import Chunk
from rag_agent.chunking.sentence_chunker import SentenceChunker
from rag_agent.ingestion.document import Document


def test_does_not_cut_mid_sentence():
    doc = Document(
        text="The quick brown fox jumps. It runs fast! Does it stop? No.",
        source="t.txt",
    )
    chunks = SentenceChunker(chunk_size=20).chunk(doc)

    joined = " ".join(c.text for c in chunks)
    for sentence in ["The quick brown fox jumps.", "It runs fast!", "Does it stop?", "No."]:
        assert sentence in joined
    assert all(isinstance(c, Chunk) for c in chunks)


def test_does_not_merge_across_paragraph_break():
    doc = Document(
        text="First paragraph sentence.\n\nSecond paragraph sentence.",
        source="t.txt",
    )
    chunks = SentenceChunker(chunk_size=1000).chunk(doc)

    assert len(chunks) == 2
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[1].text


def test_empty_document_yields_no_chunks():
    doc = Document(text="", source="t.txt")
    assert SentenceChunker(chunk_size=50).chunk(doc) == []


def test_whitespace_only_document_yields_no_chunks():
    doc = Document(text="   \n\n   ", source="t.txt")
    assert SentenceChunker(chunk_size=50).chunk(doc) == []


def test_single_very_long_sentence_falls_back_to_fixed_size():
    long_sentence = "word " * 40 + "."
    doc = Document(text=long_sentence, source="t.txt")
    chunks = SentenceChunker(chunk_size=50).chunk(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=0)


def test_sentence_aware_avoids_mid_sentence_cut_unlike_fixed_size():
    """Comparison required by #13: sentence-aware chunking should not
    cut mid-sentence the way fixed-size chunking does on the same
    input."""
    from rag_agent.chunking.chunker import FixedSizeChunker

    doc = Document(
        text="The quick brown fox jumps. It runs fast! Does it stop? No, it keeps going.",
        source="t.txt",
    )

    fixed_chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk(doc)
    sentence_chunks = SentenceChunker(chunk_size=50).chunk(doc)

    fixed_text = [c.text for c in fixed_chunks]
    sentence_text = [c.text for c in sentence_chunks]

    # Fixed-size is expected to cut mid-sentence on this input.
    assert not any(t.strip().endswith(("?", "!", ".")) for t in fixed_text[:-1])
    # Sentence-aware never does, except for the fallback on an oversized sentence.
    assert all(t.strip().endswith(("?", "!", ".")) for t in sentence_text)
