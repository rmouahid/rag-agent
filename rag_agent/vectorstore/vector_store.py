import json
import os

import numpy as np

from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.brute_force import cosine_top_k

logger = get_logger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"


class VectorStore:
    def __init__(self, path: str = None):
        """
        Stockage en mémoire, optionnellement persisté sur disque :
        - self.vectors  : matrice numpy (N x dim)
        - self.chunks   : liste des chunks associés (id, text, source, metadata)

        path : dossier de persistance (vectors.npy + chunks.json). S'il
        contient déjà un store, il est chargé tel quel (sans recalculer
        les embeddings) ; ensuite chaque `add` y est ajouté automatiquement.
        Sans path, le store reste purement en mémoire.
        """
        self.path = path
        self._matrix = None
        self.chunks = []
        self._ids = set()

        if path and self.exists(path):
            self.load(path)

    @staticmethod
    def exists(path: str) -> bool:
        """
        True si `path` contient un store sauvegardé
        """
        return os.path.isfile(os.path.join(path, VECTORS_FILE)) and os.path.isfile(
            os.path.join(path, CHUNKS_FILE)
        )

    @property
    def vectors(self) -> np.ndarray:
        if self._matrix is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._matrix

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list, embeddings: np.ndarray):
        """
        Ajoute des Chunk et leurs embeddings au store (append), et
        persiste le store si un path a été fourni.

        Les chunks dont l'id est déjà présent sont ignorés, ce qui rend
        l'ajout idempotent (ré-ingestion de la même source).
        """
        embeddings = np.asarray(embeddings)
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings: they must match one-to-one"
            )
        if len(chunks) == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be a 2-D matrix, got shape {embeddings.shape}")
        if self._matrix is not None and embeddings.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"embedding dimension {embeddings.shape[1]} does not match the store's "
                f"dimension {self._matrix.shape[1]} (was the embedding model changed?)"
            )

        new_rows = []
        for i, chunk in enumerate(chunks):
            if chunk.id in self._ids:
                logger.debug("Chunk %s déjà présent dans le store, ignoré", chunk.id)
                continue
            self._ids.add(chunk.id)
            self.chunks.append({
                "id"       : chunk.id,
                "text"     : chunk.text,
                "source"   : chunk.source,
                "metadata" : chunk.metadata,
            })
            new_rows.append(i)

        if not new_rows:
            return

        new_vectors = embeddings[new_rows]
        self._matrix = new_vectors if self._matrix is None else np.vstack([self._matrix, new_vectors])

        if self.path:
            self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Recherche exacte des top_k chunks les plus proches via similarité
        cosine (brute-force sur toute la matrice, voir brute_force.py)
        """
        if self._matrix is None:
            return []

        indices, scores = cosine_top_k(self._matrix, query_vector, top_k)

        return [
            {**self.chunks[idx], "score": float(score)}
            for idx, score in zip(indices, scores)
        ]

    def save(self, path: str = None):
        """
        Sauvegarde le store sur disque (path par défaut : celui du constructeur)
        """
        path = path or self.path
        if not path:
            raise ValueError("No path to save the vector store to: pass one or set it in the constructor")

        os.makedirs(path, exist_ok=True)
        # Écritures atomiques : un crash en cours de save laisse l'ancien store intact
        vectors_path = os.path.join(path, VECTORS_FILE)
        with open(f"{vectors_path}.tmp", "wb") as f:
            np.save(f, self.vectors)
        os.replace(f"{vectors_path}.tmp", vectors_path)

        chunks_path = os.path.join(path, CHUNKS_FILE)
        with open(f"{chunks_path}.tmp", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        os.replace(f"{chunks_path}.tmp", chunks_path)

        logger.info("VectorStore sauvegardé dans %s (%d chunks)", path, len(self.chunks))

    def load(self, path: str = None):
        """
        Charge le store depuis le disque (remplace le contenu courant)
        """
        path = path or self.path
        if not path:
            raise ValueError("No path to load the vector store from: pass one or set it in the constructor")

        matrix = np.load(os.path.join(path, VECTORS_FILE), allow_pickle=False)
        with open(os.path.join(path, CHUNKS_FILE), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if len(chunks) != len(matrix):
            raise ValueError(
                f"Corrupt vector store at {path}: {len(matrix)} vectors but {len(chunks)} chunks"
            )

        self._matrix = matrix if len(matrix) else None
        self.chunks = chunks
        self._ids = {c["id"] for c in chunks}
        logger.info("VectorStore chargé depuis %s (%d chunks)", path, len(self.chunks))
