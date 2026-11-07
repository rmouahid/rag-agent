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


# --- shingle index ------------------------------------------------------------

import random  # noqa: E402

from rag_agent.chunking.dedup import ShingleIndex, _jaccard, _normalize, _shingles  # noqa: E402


def _brute_force_dedup(chunks, existing_texts, threshold=0.9):
    """Reference: compare every chunk to every kept text, as before the index."""
    kept_shingles = [_shingles(_normalize(t)) for t in existing_texts]
    hashes = {_normalize(t) for t in existing_texts}
    kept = []
    for chunk in chunks:
        normalized = _normalize(chunk.text)
        if normalized in hashes:
            continue
        shingles = _shingles(normalized)
        if any(_jaccard(shingles, other) >= threshold for other in kept_shingles):
            continue
        hashes.add(normalized)
        kept_shingles.append(shingles)
        kept.append(chunk)
    return kept


def test_the_index_gives_the_same_result_as_comparing_everything():
    rng = random.Random(0)
    words = [f"w{i}" for i in range(40)]
    base = [" ".join(rng.choices(words, k=30)) for _ in range(60)]
    # Near-copies (one or two words changed), exact copies, short texts and fresh texts
    variants = []
    for text in base:
        tokens = text.split()
        for _ in range(rng.randint(0, 2)):
            tokens[rng.randrange(len(tokens))] = rng.choice(words)
        variants.append(" ".join(tokens))
    pool = base + variants + ["w1 w2", "w1  w2", ""] + [" ".join(rng.choices(words, k=30)) for _ in range(40)]
    rng.shuffle(pool)
    existing, batch = pool[:80], [_chunk(t, position=i) for i, t in enumerate(pool[80:])]

    assert deduplicate_chunks(batch, existing_texts=existing) == _brute_force_dedup(batch, existing)


def test_an_index_is_extended_with_new_corpus_texts_only():
    index = ShingleIndex()
    deduplicate_chunks([], existing_texts=["a b c d e f", "g h i j k l"], index=index)
    assert index.n_texts == 2

    deduplicate_chunks([], existing_texts=["a b c d e f", "g h i j k l", "m n o p q r"], index=index)

    assert index.n_texts == 3


def test_kept_chunks_are_added_to_the_index_for_the_next_call():
    index = ShingleIndex()
    kept = deduplicate_chunks([_chunk(BASE_TEXT)], existing_texts=[], index=index)

    assert kept and index.n_texts == 1
    assert deduplicate_chunks([_chunk(BASE_TEXT, source="b")], existing_texts=[BASE_TEXT], index=index) == []


def test_an_index_larger_than_the_corpus_is_rebuilt():
    index = ShingleIndex()
    deduplicate_chunks([], existing_texts=[BASE_TEXT, "other text entirely here now"], index=index)

    assert deduplicate_chunks([_chunk(BASE_TEXT)], existing_texts=[], index=index) != []
