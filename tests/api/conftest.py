from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.pipeline import IngestionPipeline


class FakeVectorStore:
    def __init__(self):
        self.chunks = []
        self.is_ready = True


class FakeRetriever:
    def __init__(self):
        self.vector_store = FakeVectorStore()


class FakeLLM:
    model_path = "models/fake.gguf"
    is_loaded = True


class FakeAgent:
    """Stand-in for RAGAgent: no model weights, no embedding model."""

    def __init__(self):
        self.pipeline = Mock(spec=IngestionPipeline)
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
