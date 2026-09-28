from unittest.mock import MagicMock, patch

import pytest
import requests

from rag_agent.ingestion import url_guard
from rag_agent.ingestion.document import Document
from rag_agent.ingestion.url_guard import BlockedURLError
from rag_agent.ingestion.web_loader import MAX_REDIRECTS, WebLoader

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


def _mock_response(text=None, content_type="text/html; charset=utf-8"):
    response = MagicMock()
    response.text = text
    response.headers = {"Content-Type": content_type}
    response.is_redirect = False
    response.raise_for_status = MagicMock()
    return response


def _redirect(location):
    response = MagicMock()
    response.headers = {"Location": location}
    response.is_redirect = True
    return response


def test_load_returns_document_with_main_content_only():
    with patch("requests.get", return_value=_mock_response(SAMPLE_HTML)):
        result = WebLoader().load("http://example.com/article")

    assert isinstance(result, Document)
    assert result.source == "http://example.com/article"
    assert result.metadata["type"] == "web"
    assert "Real article content." in result.text
    assert "Nav links" not in result.text
    assert "Footer stuff" not in result.text


def test_non_html_content_type_raises():
    with patch("requests.get", return_value=_mock_response(content_type="application/pdf")):
        with pytest.raises(ValueError):
            WebLoader().load("http://example.com/file.pdf")


def test_request_failure_raises_connection_error():
    with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(ConnectionError):
            WebLoader().load("http://example.com/timeout")


def test_private_host_is_refused_before_any_request():
    with patch("requests.get") as get:
        with pytest.raises(BlockedURLError):
            WebLoader(allow_private_hosts=False).load("http://localhost:8000/admin")

    get.assert_not_called()


def test_redirect_to_a_private_host_is_refused():
    with patch("requests.get", return_value=_redirect("http://metadata.internal/latest/meta-data/")) as get:
        with pytest.raises(BlockedURLError):
            WebLoader(allow_private_hosts=False).load("http://example.com/innocent")

    assert get.call_count == 1  # the private hop is never requested


def test_redirects_are_followed_manually_and_relative_locations_resolved():
    responses = [_redirect("/moved"), _mock_response(SAMPLE_HTML)]

    with patch("requests.get", side_effect=responses) as get:
        result = WebLoader(allow_private_hosts=False).load("http://example.com/old")

    assert [c.args[0] for c in get.call_args_list] == ["http://example.com/old", "http://example.com/moved"]
    assert all(c.kwargs["allow_redirects"] is False for c in get.call_args_list)
    assert "Real article content." in result.text


def test_redirect_loops_are_cut():
    with patch("requests.get", return_value=_redirect("http://example.com/loop")) as get:
        with pytest.raises(ConnectionError):
            WebLoader(allow_private_hosts=False).load("http://example.com/loop")

    assert get.call_count == MAX_REDIRECTS + 1


def test_private_hosts_can_be_allowed_explicitly():
    with patch("requests.get", return_value=_mock_response(SAMPLE_HTML)):
        result = WebLoader(allow_private_hosts=True).load("http://localhost:8000/wiki")

    assert "Real article content." in result.text
