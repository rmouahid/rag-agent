import pytest

from rag_agent.evaluation.judge import LLMJudge


class FixedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.mark.parametrize(
    "reply, label",
    [
        ("SUPPORTED. Every claim is in the context.", "supported"),
        ("Unsupported: the date is not in the context.", "unsupported"),
        ("I think the answer is fine.", None),
    ],
)
def test_faithfulness_verdict_is_the_first_keyword(reply, label):
    verdict = LLMJudge(FixedLLM(reply)).faithfulness("Q?", "A.", ["context"])

    assert verdict.label == label
    assert verdict.reason == reply


@pytest.mark.parametrize(
    "reply, label",
    [("CORRECT, same facts.", "correct"), ("PARTIAL: misses the year.", "partial"), ("INCORRECT", "incorrect")],
)
def test_correctness_verdict_reads_the_three_grades(reply, label):
    assert LLMJudge(FixedLLM(reply)).correctness("Q?", "A.", "Reference.").label == label


def test_the_prompt_carries_question_answer_and_context():
    llm = FixedLLM("SUPPORTED")
    LLMJudge(llm).faithfulness("When was it founded?", "In 1998.", ["Founded in 1998.", "Other chunk."])

    prompt = llm.prompts[0]
    assert "When was it founded?" in prompt
    assert "In 1998." in prompt
    assert "Founded in 1998.\n---\nOther chunk." in prompt


def test_the_context_is_truncated_to_fit_the_judge_window():
    llm = FixedLLM("SUPPORTED")
    LLMJudge(llm, max_context_chars=100).faithfulness("Q?", "A.", ["x" * 1000])

    assert "x" * 100 in llm.prompts[0]
    assert "x" * 101 not in llm.prompts[0]


def test_the_reason_is_whitespace_normalised_and_capped():
    verdict = LLMJudge(FixedLLM("SUPPORTED\n\n" + "word " * 200)).faithfulness("Q?", "A.", [])

    assert "\n" not in verdict.reason
    assert len(verdict.reason) == 300
