from rag_agent.chunking.chunker import Chunker
from rag_agent.vectorstore.embedder import Embedder
from rag_agent.vectorstore.vector_store import VectorStore
from rag_agent.config import config
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunker      = Chunker(
            chunk_size=chunk_size if chunk_size is not None else config.chunk_size,
            overlap=overlap if overlap is not None else config.chunk_overlap,
        )
        self.embedder     = Embedder()
        self.vector_store = VectorStore()

    def ingest(self, text: str, metadata: dict = {}):
        """
        Pipeline complet d'ingestion :
        Texte → Chunks → Embeddings → VectorStore
        """
        logger.info("Ingestion de : %s", metadata.get("source", "inconnu"))

        # 1. Découpe le texte en chunks
        chunks = self.chunker.chunk(text, metadata=metadata)
        logger.debug("%d chunks créés", len(chunks))

        # 2. Génère les embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_many(texts)
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

    def save(self, path: str = "vector_db"):
        self.vector_store.save(path)

    def load(self, path: str = "vector_db"):
        self.vector_store.load(path)
