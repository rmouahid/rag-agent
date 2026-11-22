from functools import partial

import httpx
import pytest

from rag_agent.ingestion import url_guard, web_loader
from rag_agent.ingestion.document import Document
from rag_agent.ingestion.url_guard import BlockedURLError
from rag_agent.ingestion.web_loader import MAX_REDIRECTS, PageTooLargeError, WebLoader

SAMPLE_HTML = """
<html><head><title>t</title><script>bad()</script></head>
<body>
<nav>Nav links</nav>
<main><h1>Title</h1><p>Real article content.</p></main>
<footer>Footer stuff</footer>
</body></html>
"""


PRIVATE_HOSTS = {"localhost": "127.0.0.1", "metadata.internal": "169.254.169.254"}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Public address for every host except PRIVATE_HOSTS; no real DNS."""

    def getaddrinfo(host, port, proto=0):
        return [(2, 1, proto, "", (PRIVATE_HOSTS.get(host, "93.184.215.14"), port))]

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", getaddrinfo)


class HTMLStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield SAMPLE_HTML.encode()


def _html():
    return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, stream=HTMLStream())


def _redirect(location):
    return httpx.Response(302, headers={"Location": location})


@pytest.fixture
def mock_http(monkeypatch):
    def install(*responses):
        calls = []
        replies = iter(responses)

        async def handle(request):
            calls.append(str(request.url))
            response = next(replies)
            if isinstance(response, Exception):
                raise response
            assert request.headers["Accept-Encoding"] == "identity"
            return response

        client = partial(httpx.AsyncClient, transport=httpx.MockTransport(handle))
        monkeypatch.setattr(web_loader.httpx, "AsyncClient", client)
        return calls

    return install


def test_load_returns_document_with_main_content_only(mock_http):
    mock_http(_html())
    result = WebLoader().load("http://example.com/article")

    assert isinstance(result, Document)
    assert result.source == "http://example.com/article"
    assert result.metadata["type"] == "web"
    assert "Real article content." in result.text
    assert "Nav links" not in result.text
    assert "Footer stuff" not in result.text


def test_non_html_content_type_raises(mock_http):
    mock_http(httpx.Response(200, headers={"Content-Type": "application/pdf"}))

    with pytest.raises(ValueError):
        WebLoader().load("http://example.com/file.pdf")


def test_request_failure_raises_connection_error(mock_http):
    mock_http(httpx.ReadTimeout("timed out"))

    with pytest.raises(ConnectionError):
        WebLoader().load("http://example.com/timeout")


def test_private_host_is_refused_before_any_request(mock_http):
    calls = mock_http()

    with pytest.raises(BlockedURLError):
        WebLoader(allow_private_hosts=False).load("http://localhost:8000/admin")

    assert calls == []


def test_redirect_to_a_private_host_is_refused(mock_http):
    calls = mock_http(_redirect("http://metadata.internal/latest/meta-data/"))

    with pytest.raises(BlockedURLError):
        WebLoader(allow_private_hosts=False).load("http://example.com/innocent")

    assert calls == ["http://example.com/innocent"]


def test_redirects_are_followed_manually_and_relative_locations_resolved(mock_http):
    calls = mock_http(_redirect("/moved"), _html())

    result = WebLoader(allow_private_hosts=False).load("http://example.com/old")

    assert calls == ["http://example.com/old", "http://example.com/moved"]
    assert "Real article content." in result.text


def test_redirect_loops_are_cut(mock_http):
    calls = mock_http(*[_redirect("/loop") for _ in range(MAX_REDIRECTS + 1)])

    with pytest.raises(ConnectionError):
        WebLoader(allow_private_hosts=False).load("http://example.com/loop")

    assert len(calls) == MAX_REDIRECTS + 1


def test_private_hosts_can_be_allowed_explicitly(mock_http):
    mock_http(_html())

    result = WebLoader(allow_private_hosts=True).load("http://localhost:8000/wiki")

    assert "Real article content." in result.text


def test_oversized_content_length_is_rejected_before_reading_body(mock_http):
    mock_http(httpx.Response(200, headers={"Content-Type": "text/html", "Content-Length": "999999999"}))

    with pytest.raises(PageTooLargeError, match="MAX_PAGE_MB"):
        WebLoader().load("http://example.com/large")
