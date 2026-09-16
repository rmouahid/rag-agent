import re

from rag_agent.chunking.chunk import Chunk
from rag_agent.chunking.chunker import FixedSizeChunker
from rag_agent.config import config
from rag_agent.ingestion.document import Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class SentenceChunker:
    """
    Découpe un Document en respectant les frontières de phrase et de
    paragraphe plutôt qu'une coupe de caractères brute (contrairement
    à FixedSizeChunker) : les phrases sont fusionnées jusqu'à
    atteindre ~chunk_size, sans jamais fusionner à travers une
    frontière de paragraphe. Une phrase seule plus longue que
    chunk_size retombe sur un découpage à taille fixe pour ce
    fragment uniquement (elle ne peut, par définition, pas être
    coupée sur une frontière de phrase).
    """

    def __init__(self, chunk_size: int = None):
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._fallback = FixedSizeChunker(chunk_size=self.chunk_size, overlap=0)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        buffer = ""
        buffer_start = None

        for paragraph, para_start in _iter_paragraphs(text):
            for sentence, sent_offset in _iter_sentences(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_start = para_start + sent_offset

                if len(sentence) > self.chunk_size:
                    if buffer:
                        chunks.append(self._make_chunk(buffer, buffer_start, document))
                        buffer, buffer_start = "", None
                    chunks.extend(self._split_oversized(sentence, sentence_start, document))
                    continue

                candidate = f"{buffer} {sentence}".strip() if buffer else sentence
                if len(candidate) > self.chunk_size and buffer:
                    chunks.append(self._make_chunk(buffer, buffer_start, document))
                    buffer, buffer_start = sentence, sentence_start
                else:
                    buffer = candidate
                    if buffer_start is None:
                        buffer_start = sentence_start

            # Ne pas fusionner de phrases à travers une frontière de paragraphe.
            if buffer:
                chunks.append(self._make_chunk(buffer, buffer_start, document))
                buffer, buffer_start = "", None

        if buffer:
            chunks.append(self._make_chunk(buffer, buffer_start, document))

        return chunks

    def _split_oversized(self, sentence: str, offset: int, document: Document) -> list[Chunk]:
        fragment_doc = Document(text=sentence, source=document.source, metadata=document.metadata)
        fragments = self._fallback.chunk(fragment_doc)
        for fragment in fragments:
            fragment.position += offset
        return fragments

    def _make_chunk(self, text: str, position: int, document: Document) -> Chunk:
        return Chunk(text=text, source=document.source, position=position, metadata=document.metadata)


def _iter_paragraphs(text: str):
    cursor = 0
    for para in _PARAGRAPH_SPLIT_RE.split(text):
        if not para:
            continue
        idx = text.index(para, cursor)
        yield para, idx
        cursor = idx + len(para)


def _iter_sentences(paragraph: str):
    cursor = 0
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        idx = paragraph.index(sentence, cursor) if sentence else cursor
        yield sentence, idx
        cursor = idx + len(sentence)
