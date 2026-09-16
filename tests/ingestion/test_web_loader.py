from unittest.mock import MagicMock, patch

import pytest
import requests

from rag_agent.ingestion.document import Document
from rag_agent.ingestion.web_loader import WebLoader

SAMPLE_HTML = """
<html><head><title>t</title><script>bad()</script></head>
<body>
<nav>Nav links</nav>
<main><h1>Title</h1><p>Real article content.</p></main>
<footer>Footer stuff</footer>
</body></html>
"""


def _mock_response(text=None, content_type="text/html; charset=utf-8"):
    response = MagicMock()
    response.text = text
    response.headers = {"Content-Type": content_type}
    response.raise_for_status = MagicMock()
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
