import math
import os

import numpy as np

from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.kmeans import assign, kmeans

logger = get_logger(__name__)

INDEX_FILE = "ivf_index.npz"


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalise chaque ligne (norme L2 = 1). Sur des vecteurs unitaires, la
    distance euclidienne est une fonction décroissante du cosinus : le
    centre le plus proche est aussi le plus similaire en cosinus. Un
    vecteur nul reste nul.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def default_n_lists(n_vectors: int) -> int:
    """
    Nombre de listes par défaut : ~sqrt(N), l'heuristique usuelle des index
    IVF (des listes de ~sqrt(N) vecteurs, soit un compromis entre le coût
    de comparaison aux centres et celui du scan des listes sondées).
    """
    return max(1, min(n_vectors, round(math.sqrt(n_vectors))))


class IVFIndex:
    """
    Index IVF (Inverted File) construit sans bibliothèque ANN.

    Un k-means sur les vecteurs normalisés donne `n_lists` centres (le
    quantificateur grossier) ; chaque vecteur est rangé dans la liste
    (posting list) de son centre le plus proche. L'index ne stocke que
    des ids de lignes : les vecteurs eux-mêmes restent dans le
    VectorStore, qui les passe à la recherche.
    """

    def __init__(self, centroids: np.ndarray, lists: list[np.ndarray]):
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.lists = [np.asarray(ids, dtype=np.int64) for ids in lists]
        if len(self.lists) != len(self.centroids):
            raise ValueError(f"{len(self.centroids)} centroids but {len(self.lists)} posting lists")

    @property
    def n_lists(self) -> int:
        return len(self.centroids)

    def __len__(self) -> int:
        return sum(len(ids) for ids in self.lists)

    @classmethod
    def build(cls, vectors: np.ndarray, n_lists: int = None, seed: int = 0, max_iter: int = 25) -> "IVFIndex":
        """
        Construit l'index sur `vectors` (N x dim) : les ids des posting
        lists sont les numéros de ligne de cette matrice.

        n_lists : nombre de centres (défaut : ~sqrt(N), plafonné à N)
        seed    : graine du k-means, pour un index reproductible
        """
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError(f"cannot build an IVF index on vectors of shape {vectors.shape}")
        n_lists = min(n_lists or default_n_lists(len(vectors)), len(vectors))

        result = kmeans(normalize(vectors), n_clusters=n_lists, max_iter=max_iter, seed=seed)
        # Tri stable par cluster : chaque posting list est triée par id croissant
        order = np.argsort(result.labels, kind="stable")
        bounds = np.cumsum(np.bincount(result.labels, minlength=n_lists))[:-1]
        lists = np.split(order, bounds)

        logger.info(
            "Index IVF construit : %d vecteurs, %d listes (k-means : %d itérations)",
            len(vectors), n_lists, result.n_iter,
        )
        return cls(result.centroids, lists)

    def save(self, path: str) -> None:
        """
        Persiste l'index dans `path`/ivf_index.npz : les centres, et les
        posting lists concaténées avec leurs bornes (format numpy pur, sans
        pickle). Écriture atomique, comme le reste du store.
        """
        os.makedirs(path, exist_ok=True)
        sizes = np.array([len(ids) for ids in self.lists], dtype=np.int64)
        ids = np.concatenate(self.lists) if self.lists else np.empty(0, dtype=np.int64)

        index_path = os.path.join(path, INDEX_FILE)
        with open(f"{index_path}.tmp", "wb") as f:
            np.savez(f, centroids=self.centroids, ids=ids, sizes=sizes)
        os.replace(f"{index_path}.tmp", index_path)
        logger.debug("Index IVF sauvegardé : %s", index_path)

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        with np.load(os.path.join(path, INDEX_FILE), allow_pickle=False) as data:
            centroids, ids, sizes = data["centroids"], data["ids"], data["sizes"]
        if sizes.sum() != len(ids):
            raise ValueError(f"Corrupt IVF index at {path}: list sizes do not add up to the stored ids")
        return cls(centroids, np.split(ids, np.cumsum(sizes)[:-1]))

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.isfile(os.path.join(path, INDEX_FILE))

    def covers(self, n_vectors: int) -> bool:
        """
        True si l'index référence exactement les lignes 0..n_vectors-1,
        chacune une seule fois (cohérence avec le store qui le porte)
        """
        if len(self) != n_vectors:
            return False
        ids = np.concatenate(self.lists) if self.lists else np.empty(0, dtype=np.int64)
        return np.array_equal(np.sort(ids), np.arange(n_vectors))

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        """
        Numéro de liste (centre le plus proche) de chaque vecteur
        """
        labels, _ = assign(normalize(vectors), self.centroids)
        return labels
