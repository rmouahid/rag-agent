import pytest


@pytest.fixture
def openapi(client):
    return client.get("/openapi.json").json()


def test_every_route_documents_its_response_model(openapi):
    expected = {
        ("/ask", "post"): "AskResponse",
        ("/documents", "post"): "IngestResponse",
        ("/health", "get"): "HealthResponse",
    }
    for (path, method), model in expected.items():
        responses = openapi["paths"][path][method]["responses"]
        success = next(r for code, r in responses.items() if code.startswith("2"))
        assert success["content"]["application/json"]["schema"]["$ref"].endswith(f"/{model}")


def test_ask_request_schema_is_typed_and_documented(openapi):
    schema = openapi["components"]["schemas"]["AskRequest"]

    assert schema["required"] == ["question"]
    assert schema["properties"]["top_k"]["minimum"] == 1
    assert schema["properties"]["top_k"]["maximum"] == 50
    assert all(p.get("description") for p in schema["properties"].values())
    assert schema["examples"]


def test_response_fields_are_documented(openapi):
    for name in ("AskResponse", "SourceOut", "IngestResponse", "HealthResponse"):
        properties = openapi["components"]["schemas"][name]["properties"]
        assert all(p.get("description") for p in properties.values()), name


@pytest.mark.parametrize(
    "body, field",
    [
        ({"question": "q", "topk": 3}, "topk"),
        ({"question": "q", "history": [{"role": "user", "content": "x", "name": "bob"}]}, "name"),
        ({"question": 42}, "question"),
        ({"question": "q", "top_k": "five"}, "top_k"),
    ],
)
def test_malformed_requests_name_the_offending_field(client, body, field):
    response = client.post("/ask", json=body)

    assert response.status_code == 422
    assert field in response.text


def test_history_length_is_bounded(client):
    history = [{"role": "user", "content": "x"}] * 101

    assert client.post("/ask", json={"question": "q", "history": history}).status_code == 422
