import os

import pytest

from rag_agent.generation.rag_agent import UnsupportedSourceError, source_type
from rag_agent.ingestion.url_guard import BlockedURLError


@pytest.fixture
def ingested(agent):
    """Records what the fake agent was asked to ingest."""
    calls = []

    def ingest(source, source_name=None):
        # Same validation as the real agent; uploads are read while they exist
        if source_type(source) == "web":
            content = None
        else:
            with open(source, "rb") as f:
                content = f.read()
        calls.append({"source": source, "source_name": source_name, "content": content})
        return ["id1", "id2"]

    agent.ingest = ingest
    return calls


def test_uploaded_markdown_is_ingested_under_its_own_name(client, ingested):
    response = client.post("/documents", files={"file": ("notes.md", b"# Title\nBody", "text/markdown")})

    assert response.status_code == 201
    assert response.json() == {"source": "notes.md", "chunks_created": 2, "chunk_ids": ["id1", "id2"]}
    (call,) = ingested
    assert call["source_name"] == "notes.md"
    assert call["source"].endswith(".md")
    assert call["content"] == b"# Title\nBody"


def test_the_temporary_upload_is_removed(client, ingested):
    client.post("/documents", files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")})

    assert not os.path.exists(ingested[0]["source"])


def test_url_is_ingested(client, ingested):
    response = client.post("/documents", data={"url": "https://example.com/page"})

    assert response.status_code == 201
    assert response.json()["source"] == "https://example.com/page"
    assert ingested[0]["source"] == "https://example.com/page"


def test_unsupported_file_type_is_rejected(client, ingested):
    response = client.post("/documents", files={"file": ("data.xlsx", b"...", "application/octet-stream")})

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]
    assert ingested == []


@pytest.mark.parametrize("url", ["ftp://example.com/a.md", "notes.md", "/etc/passwd.md"])
def test_non_http_urls_are_rejected(client, ingested, url):
    response = client.post("/documents", data={"url": url})

    assert response.status_code == 415
    assert ingested == []


def test_unreachable_url_is_a_bad_gateway(client, agent):
    def ingest(source, source_name=None):
        raise ConnectionError("Impossible de récupérer https://down.example : timeout")

    agent.ingest = ingest

    response = client.post("/documents", data={"url": "https://down.example"})

    assert response.status_code == 502


def test_non_html_page_is_rejected(client, agent):
    def ingest(source, source_name=None):
        raise ValueError("Type de contenu non supporté : 'application/json'")

    agent.ingest = ingest

    assert client.post("/documents", data={"url": "https://example.com/api"}).status_code == 415


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"data": {"url": "https://example.com"}, "files": {"file": ("a.md", b"x", "text/markdown")}}],
)
def test_exactly_one_of_file_or_url_is_required(client, ingested, kwargs):
    response = client.post("/documents", **kwargs)

    assert response.status_code == 422
    assert ingested == []


def test_already_indexed_source_creates_no_chunks(client, agent):
    agent.ingest = lambda source, source_name=None: []

    response = client.post("/documents", files={"file": ("notes.md", b"x", "text/markdown")})

    assert response.status_code == 201
    assert response.json()["chunks_created"] == 0


def test_source_type_detection():
    assert source_type("https://example.com") == "web"
    assert source_type("Report.PDF") == "pdf"
    assert source_type("notes.markdown") == "markdown"
    with pytest.raises(UnsupportedSourceError):
        source_type("archive.zip")


def test_blocked_url_is_a_bad_request(client, agent):
    def ingest(source, source_name=None):
        raise BlockedURLError("Refusing 'http://localhost/': non-public address")

    agent.ingest = ingest

    response = client.post("/documents", data={"url": "http://localhost/"})

    assert response.status_code == 400
    assert "non-public" in response.json()["detail"]
