import pytest

from rag_agent.evaluation.dataset import Evidence, GoldenQuestion
from rag_agent.evaluation.metrics import (
    content_words,
    evidence_recall,
    is_hallucination_by_rules,
    is_refusal,
    lexical_support,
    numbers,
    reciprocal_rank,
    reference_overlap,
    unsupported_numbers,
)

CONTEXT = [
    "Lumen Rail employs 1,240 people, 410 of them drivers and conductors.",
    "The company was founded in 1998 by the regional council.",
]


def question(*quotes, category="factual"):
    return GoldenQuestion(
        id="q",
        category=category,
        question="?",
        reference_answer="",
        evidence=tuple(Evidence(source="doc.md", quote=q) for q in quotes),
    )


def test_content_words_drop_stopwords_short_words_and_digits():
    assert content_words("The company was founded in 1998 by the council") == {"company", "founded", "council"}


def test_numbers_ignore_thousands_separators_and_trailing_dots():
    assert numbers("1,240 people, 14.2 million, in 1998.") == {"1240", "14.2", "1998"}


@pytest.mark.parametrize(
    "answer",
    [
        "The documents do not mention the fare.",
        "There is no information about that in the context.",
        "I cannot answer this from the documents.",
        "Le document ne précise pas ce point.",
    ],
)
def test_refusal_wordings_are_recognised(answer):
    assert is_refusal(answer)


def test_an_answer_without_llm_call_is_a_refusal():
    assert is_refusal("Lumen Rail was founded in 1998.", answered=False)
    assert not is_refusal("Lumen Rail was founded in 1998.")


def test_evidence_recall_counts_the_quotes_found_in_the_context():
    assert evidence_recall(question("employs 1,240 people", "founded in 1998"), CONTEXT) == 1.0
    assert evidence_recall(question("employs 1,240 people", "carried 14.2 million"), CONTEXT) == 0.5
    assert evidence_recall(question(category="no_answer"), CONTEXT) == 1.0


def test_reciprocal_rank_is_one_over_the_rank_of_the_first_matching_chunk():
    assert reciprocal_rank(question("founded in 1998"), CONTEXT) == 0.5
    assert reciprocal_rank(question("carried 14.2 million"), CONTEXT) == 0.0


def test_lexical_support_is_the_share_of_answer_words_found_in_the_context():
    assert lexical_support("The company employs 1,240 people.", CONTEXT) == 1.0
    assert lexical_support("The company employs many engineers.", CONTEXT) == pytest.approx(0.5)


def test_unsupported_numbers_are_the_answer_numbers_missing_from_the_context():
    assert unsupported_numbers("It employs 1240 people since 2001.", CONTEXT) == {"2001"}


def test_reference_overlap_measures_how_much_of_the_reference_is_given():
    reference = "Lumen Rail employs 1,240 people."
    assert reference_overlap("1,240 people: Lumen Rail employs them all.", reference) == 1.0
    assert reference_overlap("Lumen Rail is large.", reference) == pytest.approx(0.4)


def test_an_invented_number_is_a_hallucination_even_with_full_lexical_support():
    assert is_hallucination_by_rules("Lumen Rail employs 1,300 people.", CONTEXT)
    assert not is_hallucination_by_rules("Lumen Rail employs 1,240 people.", CONTEXT)


def test_an_answer_mostly_absent_from_the_context_is_a_hallucination():
    assert is_hallucination_by_rules("Trains are painted purple and serve breakfast.", CONTEXT)
