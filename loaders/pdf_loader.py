import fitz  # PyMuPDF

class PDFLoader:
    def load(self, path: str) -> dict:
        """
        Charge un PDF et retourne le texte + metadata
        """
        doc = fitz.open(path)
        text = ""

        for page in doc:
            text += page.get_text()

        print(f" PDF chargé : {path} ({len(doc)} pages)")

        return {
            "text": text,
            "metadata": {
                "source": path,
                "type": "pdf",
                "pages": len(doc)
            }
        }