import requests
from bs4 import BeautifulSoup

class WebLoader:
    def load(self, url: str) -> dict:
        """
        Charge une page web, extrait le texte propre + metadata
        """
        print(f" Chargement de : {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Supprime les balises inutiles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        print(f"    Page chargée ({len(text)} caractères)")

        return {
            "text": text,
            "metadata": {
                "source": url,
                "type": "web"
            }
        }