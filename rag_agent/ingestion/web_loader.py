import requests
from bs4 import BeautifulSoup

from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class WebLoader:
    def load(self, url: str) -> dict:
        """
        Charge une page web, extrait le texte propre + metadata
        """
        logger.info("Chargement de : %s", url)

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Supprime les balises inutiles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        logger.info("Page chargée (%d caractères)", len(text))

        return {
            "text": text,
            "metadata": {
                "source": url,
                "type": "web"
            }
        }
