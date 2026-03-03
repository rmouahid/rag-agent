from core.chunker import Chunker
from core.embedder import Embedder
from core.vector_store import VectorStore

class Retriever:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunker      = Chunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder     = Embedder()
        self.vector_store = VectorStore()

    def ingest(self, text: str, metadata: dict = {}):
        """
        Pipeline complet d'ingestion :
        Texte → Chunks → Embeddings → VectorStore
        """
        print(f"📄 Ingestion de : {metadata.get('source', 'inconnu')}")

        # 1. Découpe le texte en chunks
        chunks = self.chunker.chunk(text, metadata=metadata)
        print(f"   ✂️  {len(chunks)} chunks créés")

        # 2. Génère les embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_many(texts)
        print(f"   🔢 {len(embeddings)} embeddings générés")

        # 3. Stocke dans le vector store
        self.vector_store.add(chunks, embeddings)
        print(f"   ✅ Chunks indexés dans le vector store")

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