import fitz
import pytest

from rag_agent.ingestion.document import Document
from rag_agent.ingestion.pdf_loader import PDFLoader


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "First page content.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Second page content.")
    doc.save(str(path))
    doc.close()
    return str(path)


def test_load_returns_document(sample_pdf):
    result = PDFLoader().load(sample_pdf)

    assert isinstance(result, Document)
    assert result.source == sample_pdf
    assert "First page content." in result.text
    assert "Second page content." in result.text


def test_load_records_page_metadata(sample_pdf):
    result = PDFLoader().load(sample_pdf)

    assert result.metadata["type"] == "pdf"
    assert result.metadata["pages"] == 2
    assert len(result.metadata["page_char_offsets"]) == 2
    assert result.metadata["page_char_offsets"][0] == 0
