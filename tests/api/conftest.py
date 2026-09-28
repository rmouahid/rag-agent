import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app


class FakeVectorStore:
    def __init__(self):
        self.chunks = []


class FakeRetriever:
    def __init__(self):
        self.vector_store = FakeVectorStore()


class FakeLLM:
    model_path = "models/fake.gguf"


class FakeAgent:
    """Stand-in for RAGAgent: no model weights, no embedding model."""

    def __init__(self):
        self.llm = FakeLLM()
        self.retriever = FakeRetriever()


@pytest.fixture
def agent():
    return FakeAgent()


@pytest.fixture
def client(agent):
    # The context manager runs the lifespan, as uvicorn would
    with TestClient(create_app(agent_factory=lambda: agent)) as client:
        yield client
