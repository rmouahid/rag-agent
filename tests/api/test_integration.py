"""
End-to-end API tests: the real RAGAgent (loaders, chunking, deduplication,
vector store, relevance filter, prompt fitting) behind the real app, with
the stub LLM and offline embeddings of tests/conftest.py, so the suite
runs anywhere, CI included.
"""

import json

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.generation.rag_agent import RAGAgent


@pytest.fixture
def store_dir(vector_store_dir):
    return vector_store_dir


@pytest.fixture
def llm(stub_llm):
    return stub_llm


def make_client(llm, raise_server_exceptions=True):
    app = create_app(agent_factory=lambda: RAGAgent(llm=llm, min_relevance_score=0.25))
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


@pytest.fixture
def client(store_dir, llm):
    with make_client(llm) as client:
        yield client


@pytest.fixture
def upload_markdown(eiffel_text):
    def upload(client, content=eiffel_text, name="eiffel.md"):
        return client.post("/documents", files={"file": (name, content.encode(), "text/markdown")})

    return upload


def events(response):
    parsed = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((lines["event"], json.loads(lines["data"])))
    return parsed


def test_a_fresh_service_is_healthy_and_empty(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["checks"] == {"llm": True, "vector_store": True}
    assert response.json()["indexed_chunks"] == 0


def test_asking_before_any_ingestion(client, llm):
    response = client.post("/ask", json={"question": "Who built the Eiffel Tower?"})

    assert response.json()["status"] == "no_documents"
    assert llm.prompts == []


def test_ingest_then_ask(client, llm, upload_markdown):
    ingested = upload_markdown(client)
    assert ingested.status_code == 201
    assert ingested.json()["chunks_created"] >= 1
    assert client.get("/health").json()["indexed_chunks"] == ingested.json()["chunks_created"]

    response = client.post("/ask", json={"question": "Who built the Eiffel Tower?"})

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["answer"] == "Gustave Eiffel's company."
    assert body["sources"][0]["source"] == "eiffel.md"
    assert body["sources"][0]["chunk_id"] in ingested.json()["chunk_ids"]
    (prompt,) = llm.prompts
    assert "Who built the Eiffel Tower?" in prompt
    assert "1889 World's Fair" in prompt


def test_pdf_upload_is_ingested(client, make_pdf):
    response = client.post(
        "/documents", files={"file": ("tower.pdf", make_pdf("The Eiffel Tower is in Paris."), "application/pdf")}
    )

    assert response.status_code == 201
    assert response.json()["chunks_created"] == 1
    answer = client.post("/ask", json={"question": "Where is the Eiffel Tower?"}).json()
    assert answer["sources"][0]["source"] == "tower.pdf"


def test_reingesting_a_source_creates_no_chunk(client, upload_markdown):
    upload_markdown(client)

    response = upload_markdown(client)

    assert response.status_code == 201
    assert response.json()["chunks_created"] == 0


def test_an_unrelated_question_is_not_sent_to_the_model(client, llm, upload_markdown):
    upload_markdown(client)

    response = client.post("/ask", json={"question": "Quantum chromodynamics gluon confinement?"})

    assert response.json()["status"] == "no_relevant_context"
    assert response.json()["sources"] == []
    assert llm.prompts == []


def test_history_reaches_the_prompt(client, llm, upload_markdown):
    upload_markdown(client)
    history = [
        {"role": "user", "content": "Tell me about the Eiffel Tower."},
        {"role": "assistant", "content": "It stands in Paris."},
    ]

    client.post("/ask", json={"question": "Who built the Eiffel Tower?", "history": history})

    assert "It stands in Paris." in llm.prompts[0]


def test_streamed_answer(client, upload_markdown):
    upload_markdown(client)

    response = client.post("/ask/stream", json={"question": "Who built the Eiffel Tower?"})

    parsed = events(response)
    assert parsed[0][0] == "sources"
    assert parsed[0][1]["sources"][0]["source"] == "eiffel.md"
    assert [data["text"] for name, data in parsed if name == "token"] == ["Gustave", " Eiffel's", " company."]
    assert parsed[-1] == ("done", {"answer": "Gustave Eiffel's company."})


def test_the_index_survives_a_restart(store_dir, llm, upload_markdown):
    with make_client(llm) as client:
        created = upload_markdown(client).json()["chunks_created"]

    with make_client(type(llm)()) as restarted:
        assert restarted.get("/health").json()["indexed_chunks"] == created
        assert restarted.post("/ask", json={"question": "Who built the Eiffel Tower?"}).json()["status"] == "answered"


@pytest.mark.parametrize(
    "files, status_code, error",
    [
        ({"file": ("sheet.xlsx", b"PK...", "application/octet-stream")}, 415, "unsupported_source"),
        ({"file": ("broken.pdf", b"not a pdf", "application/pdf")}, 422, "unreadable_file"),
        ({"file": ("latin1.md", "Tour Eiffel, été".encode("latin-1"), "text/markdown")}, 422, "unreadable_file"),
    ],
)
def test_upload_error_paths(client, files, status_code, error):
    response = client.post("/documents", files=files)

    assert response.status_code == status_code
    assert response.json()["error"] == error
    assert client.get("/health").json()["indexed_chunks"] == 0


def test_a_url_to_a_private_address_is_refused_before_any_request(client):
    response = client.post("/documents", data={"url": "http://127.0.0.1:8000/admin"})

    assert response.status_code == 400
    assert response.json()["error"] == "blocked_url"


def test_a_model_failure_is_a_500_without_details(store_dir, llm, upload_markdown):
    llm.fail = True
    with make_client(llm, raise_server_exceptions=False) as client:
        upload_markdown(client)

        response = client.post("/ask", json={"question": "Who built the Eiffel Tower?"})

    assert response.status_code == 500
    assert response.json() == {"error": "generation_failed", "detail": "The model failed to generate an answer."}


def test_a_model_failure_while_streaming_ends_with_an_error_event(store_dir, llm, upload_markdown):
    llm.fail = True
    with make_client(llm) as client:
        upload_markdown(client)

        response = client.post("/ask/stream", json={"question": "Who built the Eiffel Tower?"})

    assert events(response)[-1] == (
        "error",
        {"error": "generation_failed", "detail": "The model failed to generate an answer."},
    )


def test_a_question_that_does_not_fit_the_context_is_a_422(client, llm, upload_markdown):
    upload_markdown(client)
    llm.prompt_budget = 5

    response = client.post("/ask", json={"question": "Who built the Eiffel Tower? " * 3})

    assert response.status_code == 422
    assert response.json()["error"] == "context_overflow"
