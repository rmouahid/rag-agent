from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from rag_agent.config import config
from rag_agent.ingestion.document import Document
from rag_agent.ingestion.url_guard import check_url
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
_MAIN_CONTENT_SELECTORS = ["main", "article"]
MAX_REDIRECTS = 5


class WebLoader:
    """
    Récupère une page web et en extrait le texte principal (article),
    en écartant la navigation, les pieds de page et autres éléments de
    boilerplate.

    Par défaut, seules les adresses publiques sont contactées, à chaque
    redirection (voir url_guard) ; `allow_private_hosts` lève ce contrôle.
    """

    def __init__(self, allow_private_hosts: bool | None = None):
        self.allow_private_hosts = config.allow_private_urls if allow_private_hosts is None else allow_private_hosts

    def _fetch(self, url: str) -> requests.Response:
        """
        GET qui suit les redirections à la main pour contrôler chaque saut
        """
        for _ in range(MAX_REDIRECTS + 1):
            if not self.allow_private_hosts:
                check_url(url)
            response = requests.get(url, timeout=10, allow_redirects=False)
            if not response.is_redirect:
                return response
            url = urljoin(url, response.headers["Location"])
        raise requests.exceptions.TooManyRedirects(f"More than {MAX_REDIRECTS} redirects")

    def load(self, url: str) -> Document:
        logger.info("Chargement de : %s", url)

        try:
            response = self._fetch(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Impossible de récupérer {url} : {e}") from e

        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            raise ValueError(f"Type de contenu non supporté pour {url} : '{content_type}' (attendu : text/html)")

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(_BOILERPLATE_TAGS):
            tag.decompose()

        main_content = None
        for selector in _MAIN_CONTENT_SELECTORS:
            main_content = soup.find(selector)
            if main_content is not None:
                break

        text = (main_content or soup).get_text(separator="\n", strip=True)

        logger.info("Page chargée (%d caractères)", len(text))

        return Document(
            text=text,
            source=url,
            metadata={"type": "web"},
        )
