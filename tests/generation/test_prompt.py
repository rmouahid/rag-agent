import pytest

from rag_agent.generation.prompt import (
    CHUNK_SEPARATOR,
    EMPTY_CONTEXT,
    build_prompt,
    format_context,
    format_history,
)


def chunk(source, text, score=1.0):
    return {"source": source, "text": text, "score": score}


def test_prompt_contains_the_question_and_each_chunk_with_its_source():
    prompt = build_prompt(
        "What is the capital of France?",
        [chunk("a.md", "Paris is the capital of France."), chunk("b.pdf", "Berlin is in Germany.")],
    )

    assert "What is the capital of France?" in prompt
    assert "[Source: a.md]\nParis is the capital of France." in prompt
    assert "[Source: b.pdf]\nBerlin is in Germany." in prompt


def test_chunks_keep_their_retrieval_order():
    prompt = build_prompt("q", [chunk("first.md", "one"), chunk("second.md", "two")])

    assert prompt.index("first.md") < prompt.index("second.md")


def test_history_is_rendered_oldest_first_before_the_question():
    history = [
        {"role": "user", "content": "Who wrote it?"},
        {"role": "assistant", "content": "Victor Hugo."},
    ]

    prompt = build_prompt("When?", [chunk("a.md", "text")], history=history)

    assert prompt.index("Who wrote it?") < prompt.index("Victor Hugo.") < prompt.index("When?")


def test_no_history_section_without_history():
    assert format_history([]) == ""
    assert "HISTORIQUE" not in build_prompt("q", [chunk("a.md", "text")])


def test_empty_context_is_stated_explicitly():
    assert format_context([]) == "(aucun document pertinent)"


@pytest.mark.parametrize("count", [0, 1, 5])
def test_prompt_with_zero_one_or_many_chunks(count):
    chunks = [chunk(f"doc{i}.md", f"passage {i}") for i in range(count)]

    prompt = build_prompt("the question", chunks)

    assert prompt.count("[Source: ") == count
    assert prompt.count(CHUNK_SEPARATOR) == max(count - 1, 0)
    assert (EMPTY_CONTEXT in prompt) == (count == 0)
    for i in range(count):
        assert f"[Source: doc{i}.md]\npassage {i}" in prompt
    # The question always closes the prompt, right before the answer cue
    assert prompt.rstrip().endswith("QUESTION :\nthe question\n\nRÉPONSE :")


def test_single_chunk_has_no_separator():
    prompt = build_prompt("q", [chunk("a.md", "only")])

    assert CHUNK_SEPARATOR not in prompt


def test_chunk_text_is_inserted_verbatim():
    text = "Braces {like this} and\nnewlines are kept: 100% intact."

    assert text in build_prompt("q", [chunk("a.md", text)])


def test_unknown_history_roles_are_rendered_as_is():
    history = [{"role": "system", "content": "note"}]

    assert "system : note" in format_history(history)
