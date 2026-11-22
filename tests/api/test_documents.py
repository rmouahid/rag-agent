import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import anyio
import httpx
import pytest

from rag_agent.api.dependencies import agent_lock
from rag_agent.generation.answer import Answer
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.ingestion import web_loader
from rag_agent.ingestion.url_guard import BlockedURLError
from rag_agent.pipeline import IngestionResult, UnsupportedSourceError, source_type


@pytest.fixture
def ingested(agent):
    """Records what the fake agent was asked to ingest."""
    calls = []

    def ingest(source, source_name=None):
        # Uploads are read while they exist
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


def test_url_is_ingested(client, agent):
    agent.pipeline.ingest.return_value = IngestionResult(source="https://example.com/page", chunk_ids=["id1"])
    response = client.post("/documents", data={"url": "https://example.com/page"})

    assert response.status_code == 201
    assert response.json()["source"] == "https://example.com/page"
    assert response.json()["chunk_ids"] == ["id1"]
    agent.pipeline.ingest.assert_called_once_with("https://example.com/page", index_lock=agent_lock)


def test_unsupported_file_type_is_rejected(client, ingested):
    response = client.post("/documents", files={"file": ("data.xlsx", b"...", "application/octet-stream")})

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]
    assert ingested == []


@pytest.mark.parametrize("url", ["ftp://example.com/a.md", "notes.md", "/etc/passwd.md"])
def test_non_http_urls_are_rejected(client, agent, url):
    response = client.post("/documents", data={"url": url})

    assert response.status_code == 415
    agent.pipeline.ingest.assert_not_called()


def test_unreachable_url_is_a_bad_gateway(client, agent):
    agent.pipeline.ingest.side_effect = ConnectionError("Impossible de récupérer https://down.example : timeout")

    response = client.post("/documents", data={"url": "https://down.example"})

    assert response.status_code == 502


def test_non_html_page_is_rejected(client, agent):
    agent.pipeline.ingest.side_effect = ValueError("Type de contenu non supporté : 'application/json'")

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
    agent.pipeline.ingest.side_effect = BlockedURLError("Refusing 'http://localhost/': non-public address")

    response = client.post("/documents", data={"url": "http://localhost/"})

    assert response.status_code == 400
    assert "non-public" in response.json()["detail"]


@pytest.fixture(params=["loopback", "mock_transport"])
def remote_page(request, monkeypatch):
    started = Event()
    send_body = Event()
    stopped = Event()

    if request.param == "mock_transport":

        class PageStream(httpx.AsyncByteStream):
            def __init__(self, path):
                self.path = path

            async def __aiter__(self):
                started.set()
                while not send_body.is_set():
                    await anyio.sleep(0.01)
                if self.path == "/slow":
                    for _ in range(100):
                        yield b"x"
                        await anyio.sleep(0.03)
                else:
                    for _ in range(17):
                        yield b"x" * 65536

        async def handle(request):
            return httpx.Response(200, headers={"Content-Type": "text/html"}, stream=PageStream(request.url.path))

        client = partial(httpx.AsyncClient, transport=httpx.MockTransport(handle))
        monkeypatch.setattr(web_loader.httpx, "AsyncClient", client)
        yield "http://example.test", started, send_body
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            started.set()
            send_body.wait(timeout=3)
            try:
                if self.path == "/slow":
                    for _ in range(100):
                        self.wfile.write(b"x")
                        if stopped.wait(0.03):
                            break
                else:
                    # Close-delimited body with no advertised length.
                    self.wfile.write(b"x" * (1024 * 1024 + 1))
            except (BrokenPipeError, ConnectionResetError):
                pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError as e:
        pytest.skip(f"Loopback sockets unavailable: {e}")

    with server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", started, send_body
        finally:
            send_body.set()
            stopped.set()
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    "path, expected_status, error, detail",
    [
        ("/slow", 502, "source_unreachable", "Délai total"),
        ("/large", 413, "page_too_large", "MAX_PAGE_MB"),
    ],
)
def test_remote_download_limits_leave_agent_available(
    client, agent, monkeypatch, vector_store_dir, stub_llm, remote_page, path, expected_status, error, detail
):
    monkeypatch.setattr(
        web_loader, "config", replace(web_loader.config, allow_private_urls=True, max_page_mb=1, page_timeout_seconds=1)
    )
    agent.pipeline = RAGAgent(llm=stub_llm).pipeline
    agent.ask = lambda question, **kwargs: Answer(text="ok")
    base_url, started, send_body = remote_page

    with ThreadPoolExecutor() as executor:
        ingestion = executor.submit(client.post, "/documents", data={"url": base_url + path})
        try:
            assert started.wait(timeout=2)
            assert agent_lock.acquire(blocking=False)
            agent_lock.release()
            assert client.post("/ask", json={"question": "q"}).json()["answer"] == "ok"
            assert not ingestion.done()
        finally:
            send_body.set()

        response = ingestion.result(timeout=2)

    assert response.status_code == expected_status
    assert response.json()["error"] == error
    assert detail in response.json()["detail"]
    assert agent.pipeline.vector_store.chunks == []
    assert agent_lock.acquire(blocking=False)
    agent_lock.release()
