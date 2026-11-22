import asyncio
from urllib.parse import urljoin

import anyio
import httpx
from bs4 import BeautifulSoup

from rag_agent.config import config
from rag_agent.ingestion.document import Document
from rag_agent.ingestion.url_guard import check_url
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
_MAIN_CONTENT_SELECTORS = ["main", "article"]
MAX_REDIRECTS = 5


class PageTooLargeError(ValueError):
    """La réponse distante dépasse la taille autorisée."""


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

    async def _read_html(self, response: httpx.Response) -> tuple[bytearray, str | None]:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            raise ValueError(
                f"Type de contenu non supporté pour {response.url} : '{content_type}' (attendu : text/html)"
            )

        max_bytes = config.max_page_mb * 1024 * 1024
        too_large = f"Remote page exceeds the {config.max_page_mb} MiB limit (MAX_PAGE_MB)."
        declared = response.headers.get("Content-Length", "")
        if declared.isascii() and declared.isdigit() and int(declared) > max_bytes:
            raise PageTooLargeError(too_large)
        # Évite toute décompression avant le contrôle de la taille.
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ConnectionError(f"Encodage HTTP non supporté pour {response.url} (attendu : identity)")

        body = bytearray()
        async for chunk in response.aiter_raw():
            if len(body) + len(chunk) > max_bytes:
                raise PageTooLargeError(too_large)
            body.extend(chunk)
            await anyio.lowlevel.checkpoint()
        return body, response.charset_encoding

    async def _fetch(self, url: str) -> tuple[bytearray, str | None]:
        async with httpx.AsyncClient(headers={"Accept-Encoding": "identity"}, timeout=None) as client:
            with anyio.fail_after(config.page_timeout_seconds):
                for _ in range(MAX_REDIRECTS + 1):
                    if not self.allow_private_hosts:
                        await anyio.to_thread.run_sync(check_url, url, abandon_on_cancel=True)
                    async with client.stream("GET", url) as response:
                        if not response.has_redirect_location:
                            return await self._read_html(response)
                        url = urljoin(url, response.headers["Location"])
        raise ConnectionError(f"Impossible de récupérer {url} : plus de {MAX_REDIRECTS} redirections")

    def load(self, url: str) -> Document:
        logger.info("Chargement de : %s", url)

        # Ferme la boucle sans attendre une résolution DNS système encore
        # bloquée dans son executor après l'annulation du téléchargement.
        loop = asyncio.new_event_loop()
        try:
            body, charset = loop.run_until_complete(self._fetch(url))
        except TimeoutError as e:
            raise ConnectionError(
                f"Délai total de téléchargement dépassé pour {url} ({config.page_timeout_seconds:g} s)"
            ) from e
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            raise ConnectionError(f"Impossible de récupérer {url} : {e}") from e
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

        soup = BeautifulSoup(bytes(body), "html.parser", from_encoding=charset)

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
