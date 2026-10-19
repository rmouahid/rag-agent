import os

import numpy as np

from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.ivf import IVFIndex, normalize
from rag_agent.vectorstore.kmeans import assign, kmeans

logger = get_logger(__name__)

INDEX_FILE = "pq_index.npz"


class ProductQuantizer:
    """
    Quantification par produit (PQ, Jégou et al., 2011), sans bibliothèque ANN.

    Chaque vecteur de dimension `dim` est découpé en `m` sous-vecteurs de
    dimension dim / m. Un k-means par sous-espace apprend un dictionnaire
    (codebook) de `ksub` = 2^n_bits centres ; un vecteur est alors stocké
    sous la forme de `m` numéros de centres (un octet chacun pour
    n_bits = 8) au lieu de `dim` flottants.

    Pour 384 dimensions en float32 (1 536 octets) et m = 16, un vecteur
    tient en 16 octets : 96 fois moins de mémoire, au prix d'une distance
    approximative.
    """

    def __init__(self, codebooks: np.ndarray):
        codebooks = np.asarray(codebooks, dtype=np.float64)
        if codebooks.ndim != 3:
            raise ValueError(f"codebooks must have shape (m, ksub, dsub), got {codebooks.shape}")
        self.codebooks = codebooks

    @property
    def m(self) -> int:
        return self.codebooks.shape[0]

    @property
    def ksub(self) -> int:
        return self.codebooks.shape[1]

    @property
    def dsub(self) -> int:
        return self.codebooks.shape[2]

    @property
    def dim(self) -> int:
        return self.m * self.dsub

    @property
    def code_dtype(self) -> type[np.unsignedinteger]:
        return np.uint8 if self.ksub <= 256 else np.uint16

    @property
    def code_size(self) -> int:
        """Octets occupés par le code d'un vecteur"""
        return self.m * np.dtype(self.code_dtype).itemsize

    @classmethod
    def train(
        cls, vectors: np.ndarray, m: int = 16, n_bits: int = 8, seed: int = 0, max_iter: int = 25
    ) -> "ProductQuantizer":
        """
        Apprend un codebook de 2^n_bits centres par sous-espace (k-means).

        m      : nombre de sous-vecteurs ; doit diviser la dimension
        n_bits : bits par sous-vecteur (1 à 16) ; 8 = un octet par sous-vecteur
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError(f"cannot train a product quantizer on vectors of shape {vectors.shape}")
        dim = vectors.shape[1]
        if m < 1 or dim % m:
            raise ValueError(f"m ({m}) must be a positive divisor of the vector dimension ({dim})")
        if not 1 <= n_bits <= 16:
            raise ValueError(f"n_bits must be between 1 and 16, got {n_bits}")
        ksub = 2**n_bits
        if ksub > len(vectors):
            raise ValueError(f"{ksub} centroids per subspace (n_bits={n_bits}) need at least {ksub} training vectors")

        dsub = dim // m
        codebooks = np.empty((m, ksub, dsub))
        for j in range(m):
            sub = vectors[:, j * dsub : (j + 1) * dsub]
            codebooks[j] = kmeans(sub, n_clusters=ksub, max_iter=max_iter, seed=seed + j).centroids
        logger.debug("PQ entraîné : %d sous-espaces de %d dimensions, %d centres chacun", m, dsub, ksub)
        return cls(codebooks)

    def _check_dim(self, vectors: np.ndarray) -> None:
        if vectors.shape[-1] != self.dim:
            raise ValueError(f"vectors of dimension {vectors.shape[-1]} do not match the quantizer ({self.dim})")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Code (N x m) : numéro du centre le plus proche dans chaque sous-espace"""
        vectors = np.asarray(vectors, dtype=np.float64)
        self._check_dim(vectors)
        codes = np.empty((len(vectors), self.m), dtype=self.code_dtype)
        for j in range(self.m):
            codes[:, j], _ = assign(vectors[:, j * self.dsub : (j + 1) * self.dsub], self.codebooks[j])
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruction approximative (N x dim) : concaténation des centres"""
        codes = np.asarray(codes)
        return np.concatenate([self.codebooks[j][codes[:, j]] for j in range(self.m)], axis=1)

    def inner_product_table(self, query: np.ndarray) -> np.ndarray:
        """
        Table (m x ksub) des produits scalaires entre chaque sous-vecteur de
        la requête et chaque centre de son sous-espace : calculée une fois
        par requête, elle suffit ensuite à scorer n'importe quel code.
        """
        query = np.asarray(query, dtype=np.float64)
        self._check_dim(query)
        return np.einsum("jd,jkd->jk", query.reshape(self.m, self.dsub), self.codebooks)

    def adc_scores(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        Calcul asymétrique de distance (ADC) : produit scalaire entre la
        requête exacte et chaque vecteur reconstruit, obtenu par m lectures
        dans la table au lieu de `dim` multiplications.
        """
        return table[np.arange(self.m), codes].sum(axis=1)


class PQIndex:
    """
    Index de recherche sur codes PQ : seuls les codes (N x m octets) et les
    codebooks sont gardés en mémoire, pas les vecteurs.

    Les vecteurs sont normalisés avant l'encodage, si bien que le produit
    scalaire approché par ADC estime la similarité cosine.

    Avec `n_lists`, un quantificateur grossier IVF (ivf.py) restreint la
    recherche aux `nprobe` listes les plus proches de la requête (IVF+PQ) ;
    sans, chaque requête score tous les codes.
    """

    def __init__(self, pq: ProductQuantizer, codes: np.ndarray, ivf: IVFIndex | None = None):
        self.pq = pq
        self.codes = np.asarray(codes, dtype=pq.code_dtype)
        self.ivf = ivf
        if self.codes.ndim != 2 or self.codes.shape[1] != pq.m:
            raise ValueError(f"codes of shape {self.codes.shape} do not match the quantizer (m={pq.m})")

    def __len__(self) -> int:
        return len(self.codes)

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        m: int = 16,
        n_bits: int = 8,
        n_lists: int | None = None,
        seed: int = 0,
        max_train: int | None = 50_000,
    ) -> "PQIndex":
        """
        Entraîne le quantificateur et encode `vectors` (ids = numéros de ligne).

        n_lists   : nombre de listes du quantificateur grossier IVF ; None =
                    pas d'IVF (scan de tous les codes)
        max_train : nombre maximal de vecteurs (tirés au hasard) utilisés pour
                    apprendre les codebooks ; tous sont ensuite encodés
        """
        unit = normalize(vectors)
        if unit.ndim != 2 or len(unit) == 0:
            raise ValueError(f"cannot build a PQ index on vectors of shape {unit.shape}")
        train = unit
        if max_train and len(unit) > max_train:
            train = unit[np.random.default_rng(seed).choice(len(unit), size=max_train, replace=False)]

        pq = ProductQuantizer.train(train, m=m, n_bits=n_bits, seed=seed)
        ivf = IVFIndex.build(unit, n_lists=n_lists, seed=seed) if n_lists else None
        index = cls(pq, pq.encode(unit), ivf)
        logger.info(
            "Index PQ construit : %d vecteurs, %d octets par code%s",
            len(index),
            pq.code_size,
            f", IVF {ivf.n_lists} listes" if ivf else "",
        )
        return index

    @property
    def memory_bytes(self) -> int:
        """Mémoire de l'index : codes, codebooks et, le cas échéant, l'IVF"""
        total = self.codes.nbytes + self.pq.codebooks.nbytes
        if self.ivf is not None:
            total += self.ivf.centroids.nbytes + sum(ids.nbytes for ids in self.ivf.lists)
        return total

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Encode de nouveaux vecteurs avec les codebooks existants (pas de
        nouvel apprentissage). ids : lignes qui prolongent l'index.
        """
        ids = np.asarray(ids, dtype=np.int64)
        expected = np.arange(len(self), len(self) + len(ids))
        if len(vectors) != len(ids) or not np.array_equal(ids, expected):
            raise ValueError("ids must match the vectors one-to-one and extend the index")
        if len(ids) == 0:
            return
        unit = normalize(vectors)
        self.codes = np.vstack([self.codes, self.pq.encode(unit)])
        if self.ivf is not None:
            self.ivf.add(unit, ids)

    def search(self, query: np.ndarray, k: int, nprobe: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        k plus proches voisins approchés en cosine. Avec l'IVF, seules les
        `nprobe` listes les plus proches sont scorées (défaut : toutes).

        Retourne (indices, scores ADC) par score décroissant, les égalités
        départagées par le plus petit id, comme la baseline exacte.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query = normalize(np.asarray(query))
        table = self.pq.inner_product_table(query)

        if self.ivf is None:
            candidates = np.arange(len(self))
        else:
            probed = self.ivf.probe(query, nprobe or self.ivf.n_lists)
            candidates = np.sort(np.concatenate([self.ivf.lists[i] for i in probed]))
        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        scores = self.pq.adc_scores(table, self.codes[candidates])
        n = len(scores)
        k = min(k, n)
        if k == n:
            top = np.arange(n)
        else:
            # Seuil du k-ième score puis tous ses ex æquo, comme cosine_top_k
            top = np.flatnonzero(scores >= np.partition(scores, n - k)[n - k])
        order = np.lexsort((candidates[top], -scores[top]))[:k]
        best = top[order]
        return candidates[best], scores[best]

    def save(self, path: str) -> None:
        """Persiste codebooks, codes et IVF éventuel (numpy pur, écriture atomique)"""
        os.makedirs(path, exist_ok=True)
        index_path = os.path.join(path, INDEX_FILE)
        with open(f"{index_path}.tmp", "wb") as f:
            np.savez(f, codebooks=self.pq.codebooks, codes=self.codes)
        os.replace(f"{index_path}.tmp", index_path)
        if self.ivf is not None:
            self.ivf.save(path)

    @classmethod
    def load(cls, path: str) -> "PQIndex":
        with np.load(os.path.join(path, INDEX_FILE), allow_pickle=False) as data:
            codebooks, codes = data["codebooks"], data["codes"]
        ivf = IVFIndex.load(path) if IVFIndex.exists(path) else None
        return cls(ProductQuantizer(codebooks), codes, ivf)
