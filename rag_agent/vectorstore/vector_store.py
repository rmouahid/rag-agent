import json
import os

import numpy as np

from rag_agent.chunking.chunk import Chunk
from rag_agent.config import config
from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.hnsw import HNSWIndex
from rag_agent.vectorstore.ivf import IVFIndex

logger = get_logger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
INDEX_TYPES = ("brute-force", "ivf", "hnsw")
INDEX_CLASSES: dict[str, type[IVFIndex] | type[HNSWIndex]] = {"ivf": IVFIndex, "hnsw": HNSWIndex}


class VectorStore:
    def __init__(
        self,
        path: str | None = None,
        index_type: str | None = None,
        n_lists: int | None = None,
        nprobe: int | None = None,
        rebuild_threshold: float | None = None,
        ivf_min_vectors: int | None = None,
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        hnsw_ef_search: int | None = None,
    ):
        """
        Stockage en mémoire, optionnellement persisté sur disque :
        - self.vectors  : matrice numpy (N x dim)
        - self.chunks   : liste des chunks associés (id, text, source, metadata)

        path : dossier de persistance (vectors.npy + chunks.json). S'il
        contient déjà un store, il est chargé tel quel (sans recalculer
        les embeddings) ; ensuite chaque `add` y est ajouté automatiquement.
        Sans path, le store reste purement en mémoire.

        index_type : "brute-force" (scan exact), "ivf" (index IVF, voir
        ivf.py) ou "hnsw" (graphe HNSW, voir hnsw.py). Défaut : INDEX_TYPE
        de la config.
        n_lists    : nombre de listes de l'index IVF (défaut : IVF_NLIST de
        la config ; 0 = ~sqrt(N)).
        nprobe     : nombre de listes IVF sondées par requête, le réglage du
        compromis recall / vitesse (défaut : NPROBE de la config).
        rebuild_threshold : part maximale de vecteurs ajoutés incrémentalement
        dans l'index IVF avant une reconstruction complète (défaut :
        IVF_REBUILD_THRESHOLD de la config ; voir `add`).
        ivf_min_vectors : en dessous de ce nombre de vecteurs, un store "ivf"
        ou "hnsw" cherche en brute-force (exact, et assez rapide à cette
        taille) et ne construit pas d'index ; il est construit au premier
        `add` qui atteint le seuil (défaut : IVF_MIN_VECTORS de la config).
        hnsw_m, hnsw_ef_construction, hnsw_ef_search : réglages du graphe
        HNSW (défaut : HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH).
        """
        index_type = index_type or config.index_type
        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}")

        self.path = path
        self.index_type = index_type
        self.n_lists = config.ivf_n_lists if n_lists is None else n_lists
        self.nprobe = nprobe or config.nprobe
        self.rebuild_threshold = config.ivf_rebuild_threshold if rebuild_threshold is None else rebuild_threshold
        self.ivf_min_vectors = config.ivf_min_vectors if ivf_min_vectors is None else ivf_min_vectors
        self.hnsw_m = hnsw_m or config.hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction or config.hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search or config.hnsw_ef_search
        self.index: IVFIndex | HNSWIndex | None = None
        self._matrix: np.ndarray | None = None
        self.chunks: list[dict] = []
        self._ids: set[str] = set()

        if path and self.exists(path):
            self.load(path)

    @staticmethod
    def exists(path: str) -> bool:
        """
        True si `path` contient un store sauvegardé
        """
        return os.path.isfile(os.path.join(path, VECTORS_FILE)) and os.path.isfile(os.path.join(path, CHUNKS_FILE))

    @property
    def vectors(self) -> np.ndarray:
        if self._matrix is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._matrix

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def uses_index(self) -> bool:
        """
        Recherche par un index approximatif : store "ivf" ou "hnsw" d'au
        moins `ivf_min_vectors` vecteurs. En dessous, le surcoût de l'index
        (construction, perte de recall) ne vaut pas le gain sur un scan exact
        """
        return (
            self.index_type in INDEX_CLASSES and self._matrix is not None and len(self._matrix) >= self.ivf_min_vectors
        )

    @property
    def is_ready(self) -> bool:
        """
        Prêt à être interrogé : un vecteur par chunk, et l'index construit
        quand la recherche passe par lui
        """
        if len(self.vectors) != len(self.chunks):
            return False
        return not self.uses_index or self.index is not None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
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
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings: they must match one-to-one")
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
            self.chunks.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                }
            )
            new_rows.append(i)

        if not new_rows:
            return

        new_vectors = embeddings[new_rows]
        first_id = 0 if self._matrix is None else len(self._matrix)
        self._matrix = new_vectors if self._matrix is None else np.vstack([self._matrix, new_vectors])

        if self.uses_index:
            self._update_index(new_vectors, first_id)

        if self.path:
            self.save()

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, nprobe: int | None = None, ef_search: int | None = None
    ) -> list[dict]:
        """
        Recherche des top_k chunks les plus proches en similarité cosine :
        - brute-force : scan exact de toute la matrice (brute_force.py)
        - ivf         : scan des seules `nprobe` listes les plus proches
                        (ivf.py) ; `nprobe` surcharge celui du store
        - hnsw        : parcours du graphe en largeur `ef_search` (hnsw.py) ;
                        `ef_search` surcharge celui du store
        """
        if self._matrix is None:
            return []

        if self.uses_index and isinstance(self.index, HNSWIndex):
            indices, scores = self.index.search(self._matrix, query_vector, top_k, ef_search or self.hnsw_ef_search)
        elif self.uses_index and isinstance(self.index, IVFIndex):
            indices, scores = self.index.search(self._matrix, query_vector, top_k, nprobe or self.nprobe)
        else:
            indices, scores = cosine_top_k(self._matrix, query_vector, top_k)

        return [{**self.chunks[idx], "score": float(score)} for idx, score in zip(indices, scores)]

    def save(self, path: str | None = None) -> None:
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

    def load(self, path: str | None = None) -> None:
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
            raise ValueError(f"Corrupt vector store at {path}: {len(matrix)} vectors but {len(chunks)} chunks")

        self._matrix = matrix if len(matrix) else None
        self.chunks = chunks
        self._ids = {c["id"] for c in chunks}
        logger.info("VectorStore chargé depuis %s (%d chunks)", path, len(self.chunks))

        self.index = None
        if self.uses_index:
            self._load_index(path)

    def rebuild_index(self) -> None:
        """
        (Re)construit l'index sur l'ensemble des vecteurs du store. IVF :
        nouveau k-means, nombre de listes recalculé si IVF_NLIST=0, dérive
        remise à zéro, à appeler à la main si le recall se dégrade. HNSW :
        nouveau graphe (rarement utile, le graphe grandit sans se dégrader).
        """
        if self._matrix is None:
            self.index = None
            return
        if self.index_type == "hnsw":
            self.index = HNSWIndex.build(
                self._matrix, m=self.hnsw_m, ef_construction=self.hnsw_ef_construction, ef_search=self.hnsw_ef_search
            )
        else:
            self.index = IVFIndex.build(self._matrix, n_lists=self.n_lists or None)

    def _update_index(self, new_vectors: np.ndarray, first_id: int) -> None:
        assert self._matrix is not None  # appelé après l'ajout des nouveaux vecteurs
        if self.index is None:
            logger.info(
                "%d vecteurs (seuil IVF_MIN_VECTORS = %d) : construction de l'index %s",
                len(self._matrix),
                self.ivf_min_vectors,
                self.index_type.upper(),
            )
            self.rebuild_index()
            return
        self.index.add(new_vectors, np.arange(first_id, first_id + len(new_vectors)))
        # Le graphe HNSW intègre chaque vecteur à sa place : seul IVF dérive
        if isinstance(self.index, IVFIndex) and self.index.drift > self.rebuild_threshold:
            logger.info(
                "Index IVF : %.0f%% de vecteurs ajoutés depuis la construction (seuil %.0f%%), reconstruction",
                100 * self.index.drift,
                100 * self.rebuild_threshold,
            )
            self.rebuild_index()

    def _load_index(self, path: str) -> None:
        """
        Recharge l'index persisté à côté du store ; s'il manque, est
        illisible ou ne correspond plus aux vecteurs, il est reconstruit.
        """
        assert self._matrix is not None  # appelé seulement quand un index est utilisé
        index_class = INDEX_CLASSES[self.index_type]
        name = self.index_type.upper()
        if index_class.exists(path):
            try:
                index = HNSWIndex.load(path, self._matrix) if index_class is HNSWIndex else IVFIndex.load(path)
                if index.covers(len(self.chunks)):
                    self.index = index
                    logger.info("Index %s chargé depuis %s", name, path)
                    return
                logger.warning("Index %s désynchronisé du store (%s), il sera reconstruit", name, path)
            except Exception as e:  # fichier tronqué / illisible
                logger.warning("Index %s illisible (%s), il sera reconstruit : %s", name, path, e)
        self.rebuild_index()
        if self.index is not None:
            self.index.save(path)
