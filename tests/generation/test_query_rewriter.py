import pytest

from rag_agent.generation.query_rewriter import QueryRewriter, parse_queries


class FixedLLM:
    def __init__(self, reply="", fail=False):
        self.reply = reply
        self.fail = fail
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("llama_decode returned -3")
        return self.reply


@pytest.mark.parametrize(
    "reply",
    [
        "Amber Line last train\nbicycle fare",
        "- Amber Line last train\n- bicycle fare",
        "1. Amber Line last train\n2) bicycle fare",
        'Query 1: "Amber Line last train"\nQuery 2: «bicycle fare»',
        "Queries:\n\nAmber Line last train\n\nbicycle fare\n",
    ],
)
def test_bullets_numbers_quotes_and_headings_are_stripped(reply):
    assert parse_queries(reply, limit=5) == ["Amber Line last train", "bicycle fare"]


def test_duplicates_are_dropped_and_the_limit_applies():
    assert parse_queries("fares\nFares\nrefunds\ndelays\nbikes", limit=2) == ["fares", "refunds"]


def test_the_original_question_always_comes_first():
    rewriter = QueryRewriter(FixedLLM("last train Amber Line\nbicycle fare"), max_queries=3)

    queries = rewriter.rewrite("When is the last train, and can I bring my bike?")

    assert queries == ["When is the last train, and can I bring my bike?", "last train Amber Line", "bicycle fare"]


def test_a_rewrite_identical_to_the_question_is_not_repeated():
    rewriter = QueryRewriter(FixedLLM("Who founded Lumen Rail?\nLumen Rail founder"))

    assert rewriter.rewrite("who founded lumen rail?") == ["who founded lumen rail?", "Lumen Rail founder"]


def test_at_most_max_queries_rewrites_are_kept():
    rewriter = QueryRewriter(FixedLLM("a1\na2\na3\na4"), max_queries=2)

    assert rewriter.rewrite("question") == ["question", "a1", "a2"]
    assert "at most 2 lines" in rewriter.llm.prompts[0]


def test_a_failing_model_falls_back_to_the_question():
    assert QueryRewriter(FixedLLM(fail=True)).rewrite("question") == ["question"]


def test_an_empty_reply_falls_back_to_the_question():
    assert QueryRewriter(FixedLLM("\n\n")).rewrite("question") == ["question"]


def test_the_conversation_is_given_to_resolve_follow_up_questions():
    llm = FixedLLM("bicycle fare")
    history = [
        {"role": "user", "content": "How much is a dog ticket?"},
        {"role": "assistant", "content": "Dogs travel for 3 euros."},
    ]

    QueryRewriter(llm).rewrite("And for bikes?", history)

    assert (
        "Conversation so far:\nuser: How much is a dog ticket?\nassistant: Dogs travel for 3 euros." in llm.prompts[0]
    )
    assert "Question: And for bikes?" in llm.prompts[0]


def test_without_history_the_prompt_has_no_conversation_block():
    llm = FixedLLM("x")
    QueryRewriter(llm).rewrite("question")

    assert "Conversation so far" not in llm.prompts[0]


def test_max_queries_must_be_positive():
    with pytest.raises(ValueError, match="max_queries"):
        QueryRewriter(FixedLLM(), max_queries=0)
