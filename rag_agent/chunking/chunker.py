from rag_agent.chunking.chunk import Chunk
from rag_agent.config import config
from rag_agent.ingestion.document import Document


class FixedSizeChunker:
    """
    Découpe un Document en chunks de longueur fixe (en caractères),
    avec chevauchement configurable. Aucun chunk produit ne dépasse
    `chunk_size` caractères.
    """

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.overlap = overlap if overlap is not None else config.chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=document.source,
                        position=start,
                        metadata=document.metadata,
                    )
                )

            start += self.chunk_size - self.overlap

        return chunks
