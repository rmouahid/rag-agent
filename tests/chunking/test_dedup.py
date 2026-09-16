from rag_agent.chunking.chunk import Chunk
from rag_agent.chunking.dedup import deduplicate_chunks

BASE_TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


def _chunk(text, source="a", position=0):
    return Chunk(text=text, source=source, position=position)


def test_exact_duplicate_within_batch_is_dropped():
    chunks = [_chunk(BASE_TEXT, source="a"), _chunk(BASE_TEXT, source="b")]
    result = deduplicate_chunks(chunks)

    assert len(result) == 1


def test_duplicate_modulo_whitespace_is_dropped():
    chunks = [_chunk(BASE_TEXT), _chunk(f"  {BASE_TEXT}  \n")]
    result = deduplicate_chunks(chunks)

    assert len(result) == 1


def test_near_duplicate_dropped_above_threshold():
    near_dup = BASE_TEXT.replace("bank.", "bank!")
    chunks = [_chunk(BASE_TEXT), _chunk(near_dup)]

    result = deduplicate_chunks(chunks, similarity_threshold=0.75)
    assert len(result) == 1


def test_near_duplicate_kept_below_threshold():
    near_dup = BASE_TEXT.replace("bank.", "bank!")
    chunks = [_chunk(BASE_TEXT), _chunk(near_dup)]

    result = deduplicate_chunks(chunks, similarity_threshold=0.95)
    assert len(result) == 2


def test_distinct_chunks_are_kept():
    chunks = [
        _chunk(BASE_TEXT),
        _chunk("Completely unrelated content about cooking pasta with tomatoes."),
    ]
    result = deduplicate_chunks(chunks)

    assert len(result) == 2


def test_dedup_against_existing_corpus():
    chunks = [_chunk(BASE_TEXT), _chunk("Completely different content here.")]
    result = deduplicate_chunks(chunks, existing_texts=[BASE_TEXT])

    assert len(result) == 1
    assert result[0].text == "Completely different content here."


def test_empty_batch_returns_empty():
    assert deduplicate_chunks([]) == []
