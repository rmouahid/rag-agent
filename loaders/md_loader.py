class MarkdownLoader:
    def load(self, path: str) -> dict:
        """
        Charge un fichier Markdown et retourne le texte + metadata
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f" Markdown chargé : {path}")

        return {
            "text": text,
            "metadata": {
                "source": path,
                "type": "markdown"
            }
        }