from rag_agent.chunking.dedup import deduplicate_chunks
from rag_agent.chunking.factory import get_chunker
from rag_agent.config import config
from rag_agent.ingestion.document import Document
from rag_agent.vectorstore.embedder import Embedder
from rag_agent.vectorstore.vector_store import VectorStore
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, chunk_size: int = None, overlap: int = None, chunking_strategy: str = None):
        self.chunker      = get_chunker(strategy=chunking_strategy, chunk_size=chunk_size, overlap=overlap)
        self.embedder     = Embedder()
        # Recharge le store existant au démarrage (VECTOR_STORE_PATH) : les
        # embeddings déjà calculés ne sont pas recalculés
        self.vector_store = VectorStore(path=config.vector_store_path or None)

    def ingest(self, document: Document):
        """
        Pipeline complet d'ingestion :
        Document → Chunks → Embeddings → VectorStore
        """
        logger.info("Ingestion de : %s", document.source)

        # 1. Découpe le document en chunks
        chunks = self.chunker.chunk(document)
        logger.debug("%d chunks créés", len(chunks))

        # 1bis. Déduplique contre le batch courant et le corpus déjà indexé
        # (ré-ingestion de la même source, ou sources qui se chevauchent)
        existing_texts = [c["text"] for c in self.vector_store.chunks]
        chunks = deduplicate_chunks(chunks, existing_texts=existing_texts)
        logger.debug("%d chunks après déduplication", len(chunks))

        if not chunks:
            logger.info("Aucun chunk nouveau après déduplication")
            return

        # 2. Génère les embeddings
        embeddings = self.embedder.embed_chunks(chunks)
        logger.debug("%d embeddings générés", len(embeddings))

        # 3. Stocke dans le vector store
        self.vector_store.add(chunks, embeddings)
        logger.info("Chunks indexés dans le vector store")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Recherche les chunks les plus pertinents pour une requête
        """
        query_vector = self.embedder.embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return results

    def save(self, path: str = None):
        self.vector_store.save(path)

    def load(self, path: str = None):
        self.vector_store.load(path)
