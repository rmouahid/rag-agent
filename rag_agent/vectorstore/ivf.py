import math
import os

import numpy as np

from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.kmeans import assign, kmeans, squared_distances

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

    Les vecteurs ajoutés après la construction sont rangés dans la liste
    du centre existant le plus proche (`add`), sans recalculer le k-means ;
    `n_added` compte ces ajouts pour décider quand reconstruire.
    """

    def __init__(self, centroids: np.ndarray, lists: list[np.ndarray], n_added: int = 0):
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.lists = [np.asarray(ids, dtype=np.int64) for ids in lists]
        self.n_added = int(n_added)
        if len(self.lists) != len(self.centroids):
            raise ValueError(f"{len(self.centroids)} centroids but {len(self.lists)} posting lists")

    @property
    def n_lists(self) -> int:
        return len(self.centroids)

    def __len__(self) -> int:
        return sum(len(ids) for ids in self.lists)

    @property
    def drift(self) -> float:
        """
        Part des vecteurs de l'index ajoutés incrémentalement depuis la
        dernière construction (0 = index fraîchement construit). Ces
        vecteurs n'ont pas participé au k-means : plus elle est grande,
        plus les centres risquent de mal représenter le corpus.
        """
        return self.n_added / len(self) if len(self) else 0.0

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
            len(vectors),
            n_lists,
            result.n_iter,
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
            np.savez(f, centroids=self.centroids, ids=ids, sizes=sizes, n_added=self.n_added)
        os.replace(f"{index_path}.tmp", index_path)
        logger.debug("Index IVF sauvegardé : %s", index_path)

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        with np.load(os.path.join(path, INDEX_FILE), allow_pickle=False) as data:
            centroids, ids, sizes = data["centroids"], data["ids"], data["sizes"]
            n_added = int(data["n_added"]) if "n_added" in data.files else 0
        if sizes.sum() != len(ids):
            raise ValueError(f"Corrupt IVF index at {path}: list sizes do not add up to the stored ids")
        return cls(centroids, np.split(ids, np.cumsum(sizes)[:-1]), n_added=n_added)

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

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Ajout incrémental : chaque vecteur est rangé dans la liste de son
        centre existant le plus proche, sans reconstruire l'index. Les
        centres ne bougent pas.

        ids : numéros de ligne des vecteurs dans la matrice du store. Ils
        doivent être supérieurs aux ids déjà indexés (ajout en fin de
        matrice), ce qui garde chaque posting list triée.
        """
        vectors = np.asarray(vectors)
        ids = np.asarray(ids, dtype=np.int64)
        if len(vectors) != len(ids):
            raise ValueError(f"{len(vectors)} vectors but {len(ids)} ids: they must match one-to-one")
        if len(ids) == 0:
            return

        labels = self.assign(vectors)
        for list_no in np.unique(labels):
            self.lists[list_no] = np.concatenate([self.lists[list_no], ids[labels == list_no]])
        self.n_added += len(ids)
        logger.debug(
            "Index IVF : %d vecteur(s) ajouté(s) sans reconstruction (dérive %.0f%%)", len(ids), 100 * self.drift
        )

    def probe(self, query: np.ndarray, nprobe: int) -> np.ndarray:
        """
        Numéros des `nprobe` listes dont le centre est le plus proche de la
        requête (même métrique que l'assignation des vecteurs aux listes)
        """
        if nprobe < 1:
            raise ValueError(f"nprobe must be >= 1, got {nprobe}")
        distances = squared_distances(normalize(query)[None, :], self.centroids)[0]
        nprobe = min(nprobe, self.n_lists)
        if nprobe == self.n_lists:
            return np.argsort(distances, kind="stable")
        nearest = np.argpartition(distances, nprobe - 1)[:nprobe]
        return nearest[np.argsort(distances[nearest], kind="stable")]

    def search(self, matrix: np.ndarray, query: np.ndarray, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Recherche approximative des k plus proches voisins en cosinus :
        on sonde les `nprobe` listes les plus proches de la requête, puis on
        fait un scan exact (cosine_top_k) sur leurs seuls vecteurs.

        matrix : la matrice de vecteurs indexée (lignes = ids des listes)
        nprobe : compromis recall / vitesse. 1 = le plus rapide ; n_lists =
                 toutes les listes, résultat identique au brute-force

        Retourne (indices, scores) triés par score décroissant, avec le même
        départage des égalités (plus petit id) que la baseline exacte. Peut
        retourner moins de k résultats si les listes sondées sont trop petites.
        """
        query = np.asarray(query)
        if query.shape != (self.centroids.shape[1],):
            raise ValueError(f"query shape {query.shape} does not match vector dimension {self.centroids.shape[1]}")

        probed = [self.lists[i] for i in self.probe(query, nprobe)]
        candidates = np.sort(np.concatenate(probed))
        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        positions, scores = cosine_top_k(matrix[candidates], query, k)
        return candidates[positions], scores
