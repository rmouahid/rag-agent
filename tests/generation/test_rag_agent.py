import pytest

from rag_agent.generation import rag_agent as rag_agent_module
from rag_agent.generation.rag_agent import RAGAgent


class FakeRetriever:
    chunks = []

    def retrieve(self, question, top_k):
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self):
        self.prompts = []

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
