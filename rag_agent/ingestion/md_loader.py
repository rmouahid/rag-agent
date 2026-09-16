from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class MarkdownLoader:
    def load(self, path: str) -> dict:
        """
        Charge un fichier Markdown et retourne le texte + metadata
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info("Markdown chargé : %s", path)

        return {
            "text": text,
            "metadata": {
                "source": path,
                "type": "markdown"
            }
        }
