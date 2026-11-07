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


def test_a_prompt_exactly_at_the_budget_is_kept_whole():
    chunks = [chunk("a.md", 30, 0.9), chunk("b.md", 30, 0.5)]
    exact = budget_for("q", chunks)

    _, kept, _ = fit_prompt("q", chunks, [], count_words, budget=exact)
    assert kept == chunks

    _, kept, _ = fit_prompt("q", chunks, [], count_words, budget=exact - 1)
    assert kept == [chunks[0]]


@pytest.mark.parametrize("count", [0, 1, 8])
def test_the_fitted_prompt_never_exceeds_the_budget(count):
    chunks = [chunk(f"{i}.md", 40, score=i / 10) for i in range(count)]
    history = [{"role": "user", "content": "past " * 20}, {"role": "assistant", "content": "reply " * 20}]
    smallest = budget_for("q", sorted(chunks, key=lambda c: c["score"])[-1:])

    for budget in range(smallest, budget_for("q", chunks, history) + 1, 7):
        prompt, kept, kept_history = fit_prompt("q", chunks, history, count_words, budget=budget)

        assert count_words(prompt) <= budget
        assert prompt == build_prompt("q", kept, history=kept_history)


def test_zero_chunks_keep_the_history_when_it_fits():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    _, kept, kept_history = fit_prompt("q", [], history, count_words, budget=budget_for("q", [], history))

    assert (kept, kept_history) == ([], history)


def test_zero_chunks_that_still_overflow_raise():
    with pytest.raises(ContextOverflowError):
        fit_prompt("long question " * 100, [], [], count_words, budget=50)


def test_many_chunks_are_dropped_one_by_one_from_the_least_relevant():
    chunks = [chunk(f"{score}.md", 20, score) for score in (0.3, 0.9, 0.1, 0.7, 0.5)]

    for keep in range(len(chunks), 0, -1):
        expected = sorted(chunks, key=lambda c: c["score"], reverse=True)[:keep]
        expected = [c for c in chunks if c in expected]

        _, kept, _ = fit_prompt("q", chunks, [], count_words, budget=budget_for("q", expected))

        assert kept == expected


def test_equal_scores_drop_the_later_chunk_first():
    first, second = chunk("first.md", 30, 0.5), chunk("second.md", 30, 0.5)

    _, kept, _ = fit_prompt("q", [first, second], [], count_words, budget=budget_for("q", [first]))

    assert kept == [first]


def test_inputs_are_not_mutated():
    chunks = [chunk("a.md", 50, 0.9), chunk("b.md", 50, 0.1)]
    history = [{"role": "user", "content": "x " * 50}, {"role": "assistant", "content": "y"}]
    before = (list(chunks), list(history))

    fit_prompt("q", chunks, history, count_words, budget=budget_for("q", chunks[:1]))

    assert (chunks, history) == before


def test_overflow_error_reports_the_sizes_and_the_setting_to_change():
    with pytest.raises(ContextOverflowError, match=r"needs \d+ tokens but only 10 .*LLM_CONTEXT_SIZE"):
        fit_prompt("q", [chunk("a.md", 100, 0.9)], [], count_words, budget=10)


def test_after_reranking_the_lowest_rerank_score_is_dropped_first():
    chunks = [
        {**chunk("a.md", 10, 0.9), "rerank_score": 0.1},
        {**chunk("b.md", 10, 0.5), "rerank_score": 4.0},
    ]

    _, kept, _ = fit_prompt("q", chunks, [], count_words, budget=budget_for("q", chunks[1:]))

    assert [c["source"] for c in kept] == ["b.md"]


def test_shrink_to_budget_drops_the_last_items_until_the_prompt_fits():
    from rag_agent.generation.truncation import shrink_to_budget

    def build(items):
        return " ".join(items)

    assert shrink_to_budget(build, ["a", "b", "c", "d"], count_words, budget=2) == "a b"
    assert shrink_to_budget(build, ["a", "b"], count_words, budget=10) == "a b"
    assert shrink_to_budget(build, ["a"], count_words, budget=0) == ""
