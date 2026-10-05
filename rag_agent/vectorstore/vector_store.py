import json
import os

import numpy as np

from rag_agent.config import config
from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.ivf import IVFIndex

logger = get_logger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
INDEX_TYPES = ("brute-force", "ivf")


class VectorStore:
    def __init__(
        self,
        path: str = None,
        index_type: str = None,
        n_lists: int = None,
        nprobe: int = None,
        rebuild_threshold: float = None,
    ):
        """
        Stockage en mémoire, optionnellement persisté sur disque :
        - self.vectors  : matrice numpy (N x dim)
        - self.chunks   : liste des chunks associés (id, text, source, metadata)

        path : dossier de persistance (vectors.npy + chunks.json). S'il
        contient déjà un store, il est chargé tel quel (sans recalculer
        les embeddings) ; ensuite chaque `add` y est ajouté automatiquement.
        Sans path, le store reste purement en mémoire.

        index_type : "brute-force" (scan exact) ou "ivf" (index IVF, voir
        ivf.py). Défaut : INDEX_TYPE de la config.
        n_lists    : nombre de listes de l'index IVF (défaut : IVF_NLIST de
        la config ; 0 = ~sqrt(N)).
        nprobe     : nombre de listes IVF sondées par requête, le réglage du
        compromis recall / vitesse (défaut : NPROBE de la config).
        rebuild_threshold : part maximale de vecteurs ajoutés incrémentalement
        dans l'index IVF avant une reconstruction complète (défaut :
        IVF_REBUILD_THRESHOLD de la config ; voir `add`).
        """
        index_type = index_type or config.index_type
        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}")

        self.path = path
        self.index_type = index_type
        self.n_lists = config.ivf_n_lists if n_lists is None else n_lists
        self.nprobe = nprobe or config.nprobe
        self.rebuild_threshold = config.ivf_rebuild_threshold if rebuild_threshold is None else rebuild_threshold
        self.index = None
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

    @property
    def is_ready(self) -> bool:
        """
        Prêt à être interrogé : un vecteur par chunk, et l'index IVF construit
        dès que le store n'est pas vide
        """
        if len(self.vectors) != len(self.chunks):
            return False
        return self.index_type != "ivf" or self._matrix is None or self.index is not None

    def add(self, chunks: list, embeddings: np.ndarray):
        """
        Ajoute des Chunk et leurs embeddings au store (append), et
        persiste le store si un path a été fourni.

        Les chunks dont l'id est déjà présent sont ignorés, ce qui rend
        l'ajout idempotent (ré-ingestion de la même source).

        Avec un index IVF, les nouveaux vecteurs sont rangés dans les listes
        existantes (pas de nouveau k-means) ; l'index n'est reconstruit que
        lorsque la part de vecteurs ajoutés ainsi dépasse `rebuild_threshold`.
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
        first_id = 0 if self._matrix is None else len(self._matrix)
        self._matrix = new_vectors if self._matrix is None else np.vstack([self._matrix, new_vectors])

        if self.index_type == "ivf":
            self._update_index(new_vectors, first_id)

        if self.path:
            self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5, nprobe: int = None) -> list[dict]:
        """
        Recherche des top_k chunks les plus proches en similarité cosine :
        - brute-force : scan exact de toute la matrice (brute_force.py)
        - ivf         : scan des seules `nprobe` listes les plus proches
                        (ivf.py) ; `nprobe` surcharge celui du store
        """
        if self._matrix is None:
            return []

        if self.index is not None:
            indices, scores = self.index.search(self._matrix, query_vector, top_k, nprobe or self.nprobe)
        else:
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

        if self.index is not None:
            self.index.save(path)

        logger.info("VectorStore sauvegardé dans %s (%d chunks)", path, len(self.chunks))

    def load(self, path: str = None):
        """
        Charge le store depuis le disque (remplace le contenu courant)
        """
        path = path or self.path
        if not path:
            raise ValueError("No path to load the vector store from: pass one or set it in the constructor")

        matrix = np.load(os.path.join(path, VECTORS_FILE), allow_pickle=False)
        with open(os.path.join(path, CHUNKS_FILE), encoding="utf-8") as f:
            chunks = json.load(f)

        if len(chunks) != len(matrix):
            raise ValueError(
                f"Corrupt vector store at {path}: {len(matrix)} vectors but {len(chunks)} chunks"
            )

        self._matrix = matrix if len(matrix) else None
        self.chunks = chunks
        self._ids = {c["id"] for c in chunks}
        logger.info("VectorStore chargé depuis %s (%d chunks)", path, len(self.chunks))

        if self.index_type == "ivf":
            self._load_index(path)

    def rebuild_index(self):
        """
        (Re)construit l'index IVF sur l'ensemble des vecteurs du store :
        nouveau k-means, nombre de listes recalculé si IVF_NLIST=0, dérive
        remise à zéro. À appeler à la main si le recall se dégrade.
        """
        if self._matrix is None:
            self.index = None
            return
        self.index = IVFIndex.build(self._matrix, n_lists=self.n_lists or None)

    def _update_index(self, new_vectors: np.ndarray, first_id: int):
        if self.index is None:
            self.rebuild_index()
            return
        self.index.add(new_vectors, np.arange(first_id, first_id + len(new_vectors)))
        if self.index.drift > self.rebuild_threshold:
            logger.info(
                "Index IVF : %.0f%% de vecteurs ajoutés depuis la construction (seuil %.0f%%), reconstruction",
                100 * self.index.drift, 100 * self.rebuild_threshold,
            )
            self.rebuild_index()

    def _load_index(self, path: str):
        """
        Recharge l'index IVF persisté à côté du store ; s'il manque, est
        illisible ou ne correspond plus aux vecteurs, il est reconstruit.
        """
        if IVFIndex.exists(path):
            try:
                index = IVFIndex.load(path)
                if index.covers(len(self.chunks)):
                    self.index = index
                    logger.info("Index IVF chargé depuis %s (%d listes)", path, index.n_lists)
                    return
                logger.warning("Index IVF désynchronisé du store (%s), il sera reconstruit", path)
            except Exception as e:  # fichier tronqué / illisible
                logger.warning("Index IVF illisible (%s), il sera reconstruit : %s", path, e)
        self.rebuild_index()
        if self.index is not None:
            self.index.save(path)
