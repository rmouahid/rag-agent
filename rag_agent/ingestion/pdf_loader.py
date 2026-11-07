import pymupdf

from rag_agent.ingestion.document import Document
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """
    Extrait le texte brut d'un PDF page par page.

    Limitation connue : `page.get_text()` lit dans l'ordre naturel des
    blocs PyMuPDF, ce qui peut entrelacer les colonnes sur des mises en
    page multi-colonnes complexes. Aucune détection de colonnes n'est
    faite ici — à traiter séparément si un cas d'usage l'exige.
    """

    def load(self, path: str) -> Document:
        doc = pymupdf.open(path)
        pages_text = [page.get_text() for page in doc.pages()]
        text = "".join(pages_text)

        logger.info("PDF chargé : %s (%d pages)", path, len(doc))

        return Document(
            text=text,
            source=path,
            metadata={
                "type": "pdf",
                "pages": len(doc),
                "page_char_offsets": _page_char_offsets(pages_text),
            },
        )


def _page_char_offsets(pages_text: list[str]) -> list[int]:
    """
    Offset (en caractères, dans le texte concaténé) où commence chaque
    page — permet de retrouver le numéro de page d'un passage donné.
    """
    offsets = []
    offset = 0
    for page_text in pages_text:
        offsets.append(offset)
        offset += len(page_text)
    return offsets
