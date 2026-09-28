import pytest

from rag_agent.generation import rag_agent as rag_agent_module
from rag_agent.generation.rag_agent import RAGAgent


class FakeRetriever:
    chunks = []

    def retrieve(self, question, top_k):
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self, prompt_budget=10_000):
        self.prompts = []
        self.prompt_budget = prompt_budget

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "answer"


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch):
    FakeRetriever.chunks = []
    monkeypatch.setattr(rag_agent_module, "Retriever", FakeRetriever)


def test_ask_sends_the_question_and_retrieved_context_to_the_local_llm():
    FakeRetriever.chunks = [
        {"source": "a.md", "text": "Paris is the capital of France."},
        {"source": "b.pdf", "text": "Berlin is the capital of Germany."},
    ]
    llm = FakeLLM()

    answer = RAGAgent(llm=llm).ask("What is the capital of France?")

    assert answer == "answer"
    (prompt,) = llm.prompts
    assert "What is the capital of France?" in prompt
    assert "[Source: a.md]\nParis is the capital of France." in prompt
    assert "[Source: b.pdf]\nBerlin is the capital of Germany." in prompt


def test_ask_without_indexed_documents_does_not_call_the_llm():
    llm = FakeLLM()

    answer = RAGAgent(llm=llm).ask("anything")

    assert "Aucun document" in answer
    assert llm.prompts == []


def test_previous_exchanges_are_recalled_in_the_next_prompt():
    FakeRetriever.chunks = [{"source": "a.md", "text": "Les Misérables, by Victor Hugo."}]
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, history_turns=3)

    agent.ask("Who wrote Les Misérables?")
    agent.ask("When was it published?")

    first, second = llm.prompts
    assert "HISTORIQUE" not in first
    assert "Who wrote Les Misérables?" in second
    assert second.index("Who wrote Les Misérables?") < second.index("When was it published?")


def test_history_keeps_only_the_last_turns():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text"}]
    agent = RAGAgent(llm=FakeLLM(), history_turns=1)

    agent.ask("first")
    agent.ask("second")

    assert [turn["content"] for turn in agent.history] == ["second", "answer"]


def test_zero_history_turns_disables_history():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text"}]
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, history_turns=0)

    agent.ask("first")
    agent.ask("second")

    assert agent.history == []
    assert "first" not in llm.prompts[1]


def test_ask_leaves_out_the_least_relevant_chunks_when_the_prompt_overflows():
    FakeRetriever.chunks = [
        {"source": "best.md", "text": "relevant " * 50, "score": 0.9},
        {"source": "worst.md", "text": "noise " * 500, "score": 0.1},
    ]
    llm = FakeLLM(prompt_budget=200)

    RAGAgent(llm=llm).ask("question")

    (prompt,) = llm.prompts
    assert "best.md" in prompt
    assert "worst.md" not in prompt
