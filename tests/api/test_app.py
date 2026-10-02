import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_agent


def test_health_reports_the_loaded_model_and_indexed_chunks(client, agent):
    agent.retriever.vector_store.chunks = [{"id": "1"}, {"id": "2"}]

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"llm": True, "vector_store": True},
        "indexed_chunks": 2,
        "model": "models/fake.gguf",
    }


@pytest.mark.parametrize(
    "component, check",
    [("llm", "llm"), ("vector_store", "vector_store")],
)
def test_health_is_503_when_a_component_is_not_ready(client, agent, component, check):
    if component == "llm":
        agent.llm.is_loaded = False
    else:
        agent.retriever.vector_store.is_ready = False

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checks"][check] is False
    assert all(ok for name, ok in body["checks"].items() if name != check)


def test_health_is_503_without_an_agent(client):
    client.app.state.agent = None

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"] == "service_unavailable"


def test_other_routes_are_503_without_an_agent(client):
    client.app.state.agent = None

    assert client.post("/ask", json={"question": "q"}).status_code == 503


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
