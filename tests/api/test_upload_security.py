"""
Ingestion path hardening (#66): request size limit and client-supplied file
names that try to escape the upload directory.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app

LIMIT = 1024


@pytest.fixture
def ingested(agent):
    calls = []

    def ingest(source, source_name=None):
        calls.append({"source": source, "source_name": source_name})
        return ["id1"]

    agent.ingest = ingest
    return calls


@pytest.fixture
def small_client(agent):
    with TestClient(create_app(agent_factory=lambda: agent, max_body_bytes=LIMIT)) as client:
        yield client


def test_an_upload_within_the_limit_is_ingested(small_client, ingested):
    response = small_client.post("/documents", files={"file": ("notes.md", b"x" * 100, "text/markdown")})

    assert response.status_code == 201
    assert len(ingested) == 1


def test_an_upload_over_the_limit_is_refused_before_ingestion(small_client, ingested):
    response = small_client.post("/documents", files={"file": ("big.md", b"x" * (LIMIT * 4), "text/markdown")})

    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"
    assert str(LIMIT) in response.json()["detail"] or "KB" in response.json()["detail"]
    assert ingested == []


def test_a_chunked_body_without_content_length_is_counted(small_client, ingested):
    boundary = "b0undary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.md"\r\n'
        "Content-Type: text/markdown\r\n\r\n"
    ).encode()

    def body():  # a generator body is sent chunked, without Content-Length
        yield head
        for _ in range(8):
            yield b"x" * 512
        yield f"\r\n--{boundary}--\r\n".encode()

    response = small_client.post(
        "/documents", content=body(), headers={"content-type": f"multipart/form-data; boundary={boundary}"}
    )

    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"
    assert ingested == []


def test_a_lying_content_length_is_not_trusted(small_client, ingested):
    # The declared length is small, the body is not: the bytes are counted anyway
    response = small_client.post(
        "/ask", content=b'{"question": "' + b"x" * (LIMIT * 2) + b'"}', headers={"content-length": "10"}
    )

    assert response.status_code in (400, 413, 422)
    assert ingested == []


def test_the_default_limit_comes_from_the_configuration(agent):
    from rag_agent.config import config

    app = create_app(agent_factory=lambda: agent)

    assert app.state.max_body_bytes == config.max_upload_mb * 1024 * 1024


@pytest.mark.parametrize(
    "filename, stored",
    [
        ("../../etc/passwd.md", "passwd.md"),
        ("/etc/cron.d/evil.md", "evil.md"),
        ("..\\..\\windows\\evil.md", "evil.md"),
        ("C:\\Users\\x\\notes.md", "notes.md"),
    ],
)
def test_a_file_name_cannot_escape_the_temporary_directory(client, ingested, filename, stored):
    response = client.post("/documents", files={"file": (filename, b"# x", "text/markdown")})

    assert response.status_code == 201
    (call,) = ingested
    assert call["source_name"] == stored
    assert response.json()["source"] == stored
    assert os.path.dirname(call["source"]) == tempfile.gettempdir()
