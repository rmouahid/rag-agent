from rag_agent.generation.prompt import build_prompt, format_context, format_history


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
