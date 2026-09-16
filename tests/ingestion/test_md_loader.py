import pytest

from rag_agent.ingestion.document import Document
from rag_agent.ingestion.md_loader import MarkdownLoader

SAMPLE_MD = """\
# Title

Some **bold** and *italic* text with a [link](http://example.com).

## Subsection

- item one
- item two

```python
def foo():
    return "**not bold**"
```

Done.
"""


@pytest.fixture
def sample_md(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    return str(path)


def test_load_returns_document(sample_md):
    result = MarkdownLoader().load(sample_md)

    assert isinstance(result, Document)
    assert result.source == sample_md


def test_headings_extracted_as_metadata(sample_md):
    result = MarkdownLoader().load(sample_md)

    assert result.metadata["headings"] == [
        {"level": 1, "text": "Title"},
        {"level": 2, "text": "Subsection"},
    ]


def test_inline_formatting_stripped(sample_md):
    result = MarkdownLoader().load(sample_md)

    assert "[link]" not in result.text
    assert "bold and italic text with a link" in result.text
    # "**" legitimately survives inside the fenced code block (preserved
    # verbatim) — only the prose line above should be formatting-free.
    prose_line = result.text.splitlines()[2]
    assert "**" not in prose_line
    assert "*italic*" not in prose_line


def test_code_block_preserved_verbatim(sample_md):
    result = MarkdownLoader().load(sample_md)

    assert '    return "**not bold**"' in result.text
