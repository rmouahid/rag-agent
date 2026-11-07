from rag_agent.generation.answer import EXCERPT_LENGTH, Answer, Source


def test_source_keeps_the_chunk_identity_and_score():
    source = Source.from_chunk({"id": "abc", "source": "a.md", "text": "Paris.", "score": 0.8})

    assert source == Source(chunk_id="abc", source="a.md", excerpt="Paris.", score=0.8)


def test_long_chunks_are_cut_to_an_excerpt_on_a_word_boundary():
    text = "word " * 100

    excerpt = Source.from_chunk({"source": "a.md", "text": text}).excerpt

    assert len(excerpt) <= EXCERPT_LENGTH + 1
    assert excerpt.endswith("word…")


def test_excerpt_collapses_whitespace():
    source = Source.from_chunk({"source": "a.md", "text": "line one\n\n  line   two"})

    assert source.excerpt == "line one line two"


def test_chunk_without_id_or_score():
    source = Source.from_chunk({"source": "a.md", "text": "text"})

    assert (source.chunk_id, source.score) == (None, None)


def test_answer_renders_its_text_and_numbered_sources():
    answer = Answer(
        text="Paris.",
        sources=[Source(chunk_id="1", source="a.md", excerpt="Paris is...", score=0.9)],
    )

    assert str(answer) == "Paris."
    assert answer.format_sources() == "[1] a.md — Paris is..."


def test_answer_without_sources_has_no_sources_section():
    assert Answer(text="Nothing.").format_sources() == ""
