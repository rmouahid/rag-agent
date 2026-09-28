import pytest

from rag_agent.generation.prompt import build_prompt
from rag_agent.generation.truncation import ContextOverflowError, fit_prompt


def count_words(text):
    return len(text.split())


def chunk(source, words, score):
    return {"source": source, "text": " ".join(["word"] * words), "score": score}


def budget_for(question, chunks, history=None):
    """Exactly the size of the prompt built from these inputs."""
    return count_words(build_prompt(question, chunks, history=history))


def test_everything_is_kept_when_the_prompt_fits():
    chunks = [chunk("a.md", 10, 0.9), chunk("b.md", 10, 0.5)]

    prompt, kept, _ = fit_prompt("q", chunks, [], count_words, budget=budget_for("q", chunks))

    assert kept == chunks
    assert prompt == build_prompt("q", chunks)


def test_lowest_relevance_chunks_are_dropped_first():
    high, low, mid = chunk("high.md", 50, 0.9), chunk("low.md", 50, 0.1), chunk("mid.md", 50, 0.5)

    prompt, kept, _ = fit_prompt("q", [high, low, mid], [], count_words, budget=budget_for("q", [high, mid]))

    assert kept == [high, mid]
    assert "low.md" not in prompt


def test_kept_chunks_stay_in_their_original_order():
    chunks = [chunk("a.md", 50, 0.2), chunk("b.md", 50, 0.1), chunk("c.md", 50, 0.9)]

    _, kept, _ = fit_prompt("q", chunks, [], count_words, budget=budget_for("q", [chunks[0], chunks[2]]))

    assert [c["source"] for c in kept] == ["a.md", "c.md"]


def test_without_scores_the_last_chunks_are_dropped_first():
    chunks = [{"source": s, "text": " ".join(["word"] * 50)} for s in ("a.md", "b.md", "c.md")]

    _, kept, _ = fit_prompt("q", chunks, [], count_words, budget=budget_for("q", chunks[:2]))

    assert [c["source"] for c in kept] == ["a.md", "b.md"]


def test_oldest_history_is_dropped_once_only_the_best_chunk_is_left():
    best, other = chunk("best.md", 20, 0.9), chunk("other.md", 20, 0.1)
    old = [{"role": "user", "content": "old " * 30}, {"role": "assistant", "content": "old " * 30}]
    recent = [{"role": "user", "content": "recent"}, {"role": "assistant", "content": "recent"}]

    prompt, kept, history = fit_prompt(
        "q", [best, other], old + recent, count_words, budget=budget_for("q", [best], recent)
    )

    assert kept == [best]
    assert history == recent
    assert "old" not in prompt


def test_overflow_with_a_single_chunk_and_no_history_raises():
    with pytest.raises(ContextOverflowError):
        fit_prompt("q", [chunk("a.md", 500, 0.9)], [], count_words, budget=100)
