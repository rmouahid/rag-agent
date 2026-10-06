"""
Pipeline d'ingestion : une source devient des chunks indexés et
cherchables, sans que l'appelant enchaîne lui-même les étapes.

    source → loader → Document → chunks → déduplication → embeddings → vector store

Le RAGAgent en possède un, partagé avec son Retriever ; la CLI et l'API
ingèrent toutes deux par lui.
"""

from dataclasses import dataclass, field

from rag_agent.chunking.chunker import FixedSizeChunker
from rag_agent.chunking.dedup import deduplicate_chunks
from rag_agent.chunking.sentence_chunker import SentenceChunker
from rag_agent.ingestion.document import Document
from rag_agent.ingestion.md_loader import MarkdownLoader
from rag_agent.ingestion.pdf_loader import PDFLoader
from rag_agent.ingestion.web_loader import WebLoader
from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.embedder import Embedder
from rag_agent.vectorstore.vector_store import VectorStore

logger = get_logger(__name__)

# Extension de fichier → loader
FILE_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown"}


class UnsupportedSourceError(ValueError):
    """Ni une URL http(s), ni un fichier d'un type pris en charge."""


def source_type(source: str) -> str:
    """
    Loader à utiliser pour `source` : "web", "pdf" ou "markdown"
    """
    if source.startswith(("http://", "https://")):
        return "web"
    for extension, kind in FILE_TYPES.items():
        if source.lower().endswith(extension):
            return kind
    raise UnsupportedSourceError(
        f"Unsupported source {source!r}: expected an http(s) URL or a {', '.join(FILE_TYPES)} file"
    )


@dataclass
class IngestionResult:
    """Bilan de l'ingestion d'une source."""

    source: str
    # Chunks nouvellement indexés (vide si la source l'était déjà)
    chunk_ids: list[str] = field(default_factory=list)
    # Chunks produits par le découpage, doublons compris
    chunks_total: int = 0

    @property
    def chunks_indexed(self) -> int:
        return len(self.chunk_ids)

    @property
    def duplicates(self) -> int:
        """Chunks écartés car déjà présents (dans la source ou dans le corpus indexé)."""
        return self.chunks_total - self.chunks_indexed


class IngestionPipeline:
    def __init__(
        self,
        chunker: FixedSizeChunker | SentenceChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        loaders: dict[str, PDFLoader | MarkdownLoader | WebLoader] | None = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.loaders = loaders or {"pdf": PDFLoader(), "markdown": MarkdownLoader(), "web": WebLoader()}

    def ingest(self, source: str, source_name: str | None = None) -> IngestionResult:
        """
        Charge `source` (fichier PDF/Markdown ou URL) avec le loader de son
        type, puis l'indexe. `source_name` remplace la source enregistrée
        avec les chunks — pour un fichier reçu par l'API, stocké sous un nom
        temporaire.
        """
        loader = self.loaders[source_type(source)]
        document = loader.load(source)
        if source_name:
            document.source = source_name
        return self.ingest_document(document)

    def ingest_document(self, document: Document) -> IngestionResult:
        """
        Document déjà chargé → chunks → déduplication → embeddings → vector store
        """
        logger.info("Ingestion de : %s", document.source)

        chunks = self.chunker.chunk(document)
        result = IngestionResult(source=document.source, chunks_total=len(chunks))
        logger.debug("%d chunks créés", len(chunks))

        # Déduplique contre le batch courant et le corpus déjà indexé
        # (ré-ingestion de la même source, ou sources qui se chevauchent)
        existing_texts = [c["text"] for c in self.vector_store.chunks]
        chunks = deduplicate_chunks(chunks, existing_texts=existing_texts)
        logger.debug("%d chunks après déduplication", len(chunks))
        if not chunks:
            logger.info("Aucun chunk nouveau après déduplication")
            return result

        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add(chunks, embeddings)
        result.chunk_ids = [c.id for c in chunks]
        logger.info("%d chunks indexés (%d doublons écartés)", result.chunks_indexed, result.duplicates)
        return result
