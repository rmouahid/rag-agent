from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_agent


def test_health_reports_the_loaded_model_and_indexed_chunks(client, agent):
    agent.retriever.vector_store.chunks = [{"id": "1"}, {"id": "2"}]

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "indexed_chunks": 2, "model": "models/fake.gguf"}


def test_the_agent_is_built_once_at_startup(agent):
    calls = []

    def factory():
        calls.append(1)
        return agent

    app = create_app(agent_factory=factory)
    assert calls == []

    with TestClient(app) as client:
        client.get("/health")
        client.get("/health")

    assert calls == [1]


def test_the_agent_dependency_can_be_overridden(agent):
    other = type(agent)()
    other.llm.model_path = "models/other.gguf"
    app = create_app(agent_factory=lambda: agent)
    app.dependency_overrides[get_agent] = lambda: other

    with TestClient(app) as client:
        assert client.get("/health").json()["model"] == "models/other.gguf"


def test_openapi_schema_lists_the_routes(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert "/health" in paths
