from rag_agent.chunking.factory import get_chunker
from rag_agent.config import config
from rag_agent.ingestion.document import Document
from rag_agent.logging_config import get_logger
from rag_agent.pipeline import IngestionPipeline
from rag_agent.vectorstore.embedder import Embedder
from rag_agent.vectorstore.vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None, chunking_strategy: str | None = None):
        self.chunker = get_chunker(strategy=chunking_strategy, chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder()
        # Recharge le store existant au démarrage (VECTOR_STORE_PATH) : les
        # embeddings déjà calculés ne sont pas recalculés
        self.vector_store = VectorStore(path=config.vector_store_path or None)
        # L'ingestion passe par les mêmes chunker, embedder et store que la recherche
        self.pipeline = IngestionPipeline(self.chunker, self.embedder, self.vector_store)

    def ingest(self, document: Document) -> list[str]:
        """
        Indexe un document déjà chargé (voir IngestionPipeline). Retourne
        les ids des chunks nouvellement indexés (vide si tout était déjà
        présent)
        """
        return self.pipeline.ingest_document(document).chunk_ids

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Recherche les chunks les plus pertinents pour une requête
        """
        query_vector = self.embedder.embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return results

    def save(self, path: str | None = None) -> None:
        self.vector_store.save(path)

    def load(self, path: str | None = None) -> None:
        self.vector_store.load(path)
