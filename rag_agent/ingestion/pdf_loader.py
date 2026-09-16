import fitz  # PyMuPDF

from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class PDFLoader:
    def load(self, path: str) -> dict:
        """
        Charge un PDF et retourne le texte + metadata
        """
        doc = fitz.open(path)
        text = ""

        for page in doc:
            text += page.get_text()

        logger.info("PDF chargé : %s (%d pages)", path, len(doc))

        return {
            "text": text,
            "metadata": {
                "source": path,
                "type": "pdf",
                "pages": len(doc)
            }
        }
