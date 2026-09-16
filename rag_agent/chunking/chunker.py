class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        chunk_size : nombre de caractères par chunk
        overlap    : chevauchement entre chunks pour ne pas perdre le contexte
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict = {}) -> list[dict]:
        """
        Découpe un texte en chunks avec overlap.
        Retourne une liste de dicts : { "text": ..., "metadata": ... }
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata
                })

            start += self.chunk_size - self.overlap

        return chunks
