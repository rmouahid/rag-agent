import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.generation.answer import Answer
from rag_agent.generation.truncation import ContextOverflowError
from rag_agent.ingestion.md_loader import MarkdownLoader
from rag_agent.ingestion.pdf_loader import PDFLoader


def assert_envelope(response, status_code, error):
    assert response.status_code == status_code
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == error
    return body["detail"]


@pytest.fixture
def lenient_client(agent):
    """Returns the 500 response instead of re-raising the server error."""
    with TestClient(create_app(agent_factory=lambda: agent), raise_server_exceptions=False) as client:
        yield client


def test_validation_errors_list_each_rejected_field(client):
    detail = assert_envelope(client.post("/ask", json={"question": "", "top_k": 0}), 422, "validation_error")

    assert {d["field"] for d in detail} == {"body.question", "body.top_k"}
    assert all(d["message"] and d["type"] for d in detail)


def test_validation_errors_do_not_echo_the_input(client):
    detail = assert_envelope(client.post("/ask", json={"question": "x" * 4001}), 422, "validation_error")

    assert "xxxx" not in str(detail)


def test_unknown_route_uses_the_envelope(client):
    assert_envelope(client.get("/nope"), 404, "not_found")


def test_wrong_method_uses_the_envelope(client):
    assert_envelope(client.get("/ask"), 405, "method_not_allowed")


def test_generation_failure_hides_the_traceback(lenient_client, agent):
    def ask(question, top_k=5, history=None):
        raise RuntimeError("llama_decode returned -3 at /home/build/llama.cpp")

    agent.ask = ask

    detail = assert_envelope(lenient_client.post("/ask", json={"question": "q"}), 500, "generation_failed")

    assert "llama" not in detail and "Traceback" not in detail


def test_ingestion_failure_hides_the_traceback(lenient_client, agent):
    def ingest(source, source_name=None):
        raise OSError("No space left on device: '/var/lib/rag/vectors.npy'")

    agent.ingest = ingest

    response = lenient_client.post("/documents", files={"file": ("notes.md", b"x", "text/markdown")})

    detail = assert_envelope(response, 500, "ingestion_failed")
    assert "notes.md" in detail and "/var/lib" not in detail


def test_ingestion_failure_of_a_url(lenient_client, agent):
    def ingest(source, source_name=None):
        raise KeyError("text")

    agent.ingest = ingest

    assert_envelope(lenient_client.post("/documents", data={"url": "https://example.com"}), 500, "ingestion_failed")


def test_unexpected_errors_outside_routes_use_the_envelope(lenient_client, agent):
    # The health route reads the vector store: break it
    agent.retriever.vector_store = None

    detail = assert_envelope(lenient_client.get("/health"), 500, "internal_error")

    assert "NoneType" not in detail


@pytest.mark.parametrize(
    "content, name",
    [(b"not a pdf", "paper.pdf"), (b"\xff\xfe\x00bad", "notes.md")],
)
def test_corrupted_upload_is_a_422(client, agent, content, name):
    # The real loaders, without embeddings: they fail while reading
    readers = {".pdf": PDFLoader(), ".md": MarkdownLoader()}
    agent.ingest = lambda source, source_name=None: readers[source[source.rfind(".") :]].load(source)

    response = client.post("/documents", files={"file": (name, content, "application/octet-stream")})

    detail = assert_envelope(response, 422, "unreadable_file")
    assert name in detail and "/tmp" not in detail


@pytest.mark.parametrize(
    "kwargs, status_code, error",
    [
        ({"files": {"file": ("data.xlsx", b"...", "application/octet-stream")}}, 415, "unsupported_source"),
        ({"data": {"url": "ftp://example.com/a.md"}}, 415, "unsupported_source"),
        ({}, 422, "invalid_source"),
    ],
)
def test_ingestion_errors_have_stable_codes(client, kwargs, status_code, error):
    assert_envelope(client.post("/documents", **kwargs), status_code, error)


def test_prompt_overflow_has_its_own_code(client, agent):
    def ask(question, top_k=5, history=None):
        raise ContextOverflowError("The prompt needs 5000 tokens")

    agent.ask = ask

    assert "5000" in assert_envelope(client.post("/ask", json={"question": "q"}), 422, "context_overflow")


def test_error_responses_are_documented(client):
    paths = client.get("/openapi.json").json()["paths"]

    for path, codes in {"/ask": ["422", "500"], "/documents": ["400", "415", "422", "500", "502"]}.items():
        responses = paths[path]["post"]["responses"]
        for code in codes:
            assert responses[code]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")


def test_successful_responses_are_unchanged(client, agent):
    agent.ask = lambda question, top_k=5, history=None: Answer(text="ok")

    assert client.post("/ask", json={"question": "q"}).json()["answer"] == "ok"
