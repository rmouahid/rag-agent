"""
Index HNSW (Hierarchical Navigable Small World) construit sans bibliothèque
ANN, d'après Malkov & Yashunin (2016).

Les vecteurs sont les nœuds d'un graphe de proximité à plusieurs couches.
Chaque nœud reçoit un niveau tiré au hasard (loi géométrique : la moitié
des nœuds environ s'arrête à la couche 0 pour m = 2, bien plus pour m = 16)
et est relié, à chaque couche jusqu'à ce niveau, à ses plus proches
voisins. Les couches hautes, très clairsemées, servent d'autoroutes : une
recherche y descend de façon gloutonne vers la bonne région, puis explore
la couche 0 en largeur `ef`.

Réglages :
- m               : voisins par nœud dans les couches hautes (2m en couche 0) ;
- ef_construction : largeur de la recherche à l'insertion (qualité du graphe) ;
- ef_search       : largeur de la recherche à la requête, le compromis
                    recall / vitesse (l'équivalent de `nprobe` pour IVF).

La similarité est le cosinus : les vecteurs sont normalisés et la distance
du graphe vaut 1 - cosinus. Les k meilleurs candidats trouvés sont ensuite
re-classés exactement (cosine_top_k), avec le même départage des égalités
que le brute-force.
"""

import heapq
import math
import os

import numpy as np

from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.brute_force import cosine_top_k
from rag_agent.vectorstore.ivf import normalize

logger = get_logger(__name__)

INDEX_FILE = "hnsw_index.npz"


class HNSWIndex:
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 100, ef_search: int = 64, seed: int = 0):
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {ef_construction}")
        if ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef_search}")
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        # Facteur de la loi des niveaux : P(niveau >= l) = m^-l
        self.level_mult = 1 / math.log(m)
        self._rng = np.random.default_rng(seed)
        # links[couche][nœud] = voisins du nœud dans cette couche
        self.links: list[dict[int, list[int]]] = []
        self.levels = np.empty(0, dtype=np.int64)
        self.entry_point: int | None = None
        # Vecteurs normalisés, en float32 : servent à naviguer dans le graphe
        self._data = np.empty((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.levels)

    def max_links(self, level: int) -> int:
        """Degré maximal d'un nœud : 2m en couche 0, la plus dense, m au-dessus."""
        return 2 * self.m if level == 0 else self.m

    @classmethod
    def build(
        cls, vectors: np.ndarray, m: int = 16, ef_construction: int = 100, ef_search: int = 64, seed: int = 0
    ) -> "HNSWIndex":
        """Construit le graphe en insérant les vecteurs un par un (ids 0..N-1)."""
        vectors = np.asarray(vectors)
        index = cls(vectors.shape[1], m=m, ef_construction=ef_construction, ef_search=ef_search, seed=seed)
        index.add(vectors, np.arange(len(vectors)))
        return index

    # --- insertion ---------------------------------------------------------------

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Insère des vecteurs dans le graphe existant, sans le reconstruire.

        ids : numéros de ligne des vecteurs dans la matrice du store ; ils
        doivent prolonger l'index (len(self), len(self) + 1, ...), comme les
        lignes ajoutées en fin de matrice.
        """
        vectors = np.asarray(vectors)
        ids = np.asarray(ids, dtype=np.int64)
        if len(vectors) != len(ids):
            raise ValueError(f"{len(vectors)} vectors but {len(ids)} ids: they must match one-to-one")
        if len(ids) == 0:
            return
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors of shape {vectors.shape} do not match the index dimension {self.dim}")
        expected = np.arange(len(self), len(self) + len(ids))
        if not np.array_equal(ids, expected):
            raise ValueError(f"ids must extend the index: expected rows {expected[0]}..{expected[-1]}")

        self._data = np.vstack([self._data, normalize(vectors).astype(np.float32)])
        # 1 - U est dans ]0, 1] : le logarithme est toujours défini
        new_levels = np.floor(-np.log(1.0 - self._rng.random(len(ids))) * self.level_mult).astype(np.int64)
        self.levels = np.concatenate([self.levels, new_levels])
        for node, level in zip(ids.tolist(), new_levels.tolist(), strict=True):
            self._insert(node, level)

    def _insert(self, node: int, level: int) -> None:
        query = self._data[node]
        while len(self.links) <= level:
            self.links.append({})
        for layer in range(level + 1):
            self.links[layer][node] = []

        if self.entry_point is None:
            self.entry_point = node
            return

        top = int(self.levels[self.entry_point])
        entry = [self.entry_point]
        # Descente gloutonne par les couches au-dessus du niveau du nœud
        for layer in range(top, level, -1):
            entry = [self._search_layer(query, entry, 1, layer)[0][1]]

        for layer in range(min(level, top), -1, -1):
            candidates = self._search_layer(query, entry, self.ef_construction, layer)
            neighbours = self._select_neighbours(candidates, self.m)
            self.links[layer][node] = neighbours
            cap = self.max_links(layer)
            for neighbour in neighbours:
                links = self.links[layer][neighbour]
                links.append(node)
                if len(links) > cap:
                    # Trop de liens : on garde les plus utiles, selon la même heuristique
                    distances = self._distances(self._data[neighbour], links)
                    ranked = sorted(zip(distances.tolist(), links, strict=True))
                    self.links[layer][neighbour] = self._select_neighbours(ranked, cap)
            entry = [candidate for _, candidate in candidates]

        if level > top:
            self.entry_point = node

    def _select_neighbours(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """
        Heuristique de sélection de l'article (algorithme 4) : un candidat
        n'est retenu que s'il est plus proche du nœud que de tous les voisins
        déjà retenus. Les liens partent ainsi dans des directions variées, ce
        qui garde le graphe navigable entre deux grappes de vecteurs proches.
        Les places restantes sont complétées par les candidats écartés.

        candidates : (distance au nœud, candidat), triés par distance croissante
        """
        selected: list[int] = []
        pruned: list[int] = []
        for distance, candidate in candidates:
            if len(selected) >= m:
                break
            if selected:
                to_selected = 1.0 - self._data[selected] @ self._data[candidate]
                if np.any(to_selected < distance):
                    pruned.append(candidate)
                    continue
            selected.append(candidate)
        return selected + pruned[: m - len(selected)]

    # --- recherche ---------------------------------------------------------------

    def _distances(self, query: np.ndarray, nodes: list[int]) -> np.ndarray:
        return 1.0 - self._data[nodes] @ query

    def _search_layer(self, query: np.ndarray, entry: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """
        Recherche « best-first » dans une couche : on explore toujours le
        candidat le plus proche non encore exploré, et on s'arrête quand il
        est plus loin que le pire des `ef` meilleurs résultats.

        Retourne au plus `ef` couples (distance, nœud) triés par distance.
        """
        visited = set(entry)
        distances = self._distances(query, entry).tolist()
        candidates = list(zip(distances, entry, strict=True))
        heapq.heapify(candidates)
        # Tas max des résultats, via les distances négatives
        results = [(-d, n) for d, n in candidates]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        links = self.links[layer]
        while candidates:
            distance, node = heapq.heappop(candidates)
            if distance > -results[0][0]:
                break
            neighbours = [n for n in links[node] if n not in visited]
            if not neighbours:
                continue
            visited.update(neighbours)
            for d, n in zip(self._distances(query, neighbours).tolist(), neighbours, strict=True):
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-d, n) for d, n in results)

    def search(self, matrix: np.ndarray, query: np.ndarray, k: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Recherche approximative des k plus proches voisins en cosinus.

        matrix : la matrice de vecteurs indexée (lignes = ids des nœuds)
        ef     : largeur de recherche en couche 0 (au moins k) ; plus elle
                 est grande, meilleur est le recall et plus la requête est lente

        Retourne (indices, scores) triés par score décroissant, re-classés
        exactement comme le brute-force.
        """
        query = np.asarray(query)
        if query.shape != (self.dim,):
            raise ValueError(f"query shape {query.shape} does not match vector dimension {self.dim}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self.entry_point is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        normalized = normalize(query).astype(np.float32)
        entry = [self.entry_point]
        for layer in range(int(self.levels[self.entry_point]), 0, -1):
            entry = [self._search_layer(normalized, entry, 1, layer)[0][1]]
        found = self._search_layer(normalized, entry, max(ef, k), 0)

        candidates = np.array(sorted(node for _, node in found), dtype=np.int64)
        positions, scores = cosine_top_k(matrix[candidates], query, k)
        return candidates[positions], scores

    # --- persistance ---------------------------------------------------------------

    def covers(self, n_vectors: int) -> bool:
        """True si le graphe contient exactement les lignes 0..n_vectors-1."""
        return len(self) == n_vectors and (not self.links or set(self.links[0]) == set(range(n_vectors)))

    def save(self, path: str) -> None:
        """
        Persiste le graphe dans `path`/hnsw_index.npz (format numpy pur, sans
        pickle) : niveaux des nœuds, point d'entrée, réglages, et pour chaque
        couche ses listes de voisins au format CSR. Les vecteurs ne sont pas
        dupliqués : ils sont relus depuis le store au chargement.
        """
        os.makedirs(path, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "levels": self.levels,
            "params": np.array(
                [
                    self.m,
                    self.ef_construction,
                    self.ef_search,
                    self.seed,
                    -1 if self.entry_point is None else self.entry_point,
                ]
            ),
            "n_layers": np.array(len(self.links)),
        }
        for layer, links in enumerate(self.links):
            nodes = np.array(sorted(links), dtype=np.int64)
            sizes = np.array([len(links[n]) for n in nodes.tolist()], dtype=np.int64)
            arrays[f"nodes_{layer}"] = nodes
            arrays[f"sizes_{layer}"] = sizes
            arrays[f"neighbours_{layer}"] = np.array(
                [n for node in nodes.tolist() for n in links[node]], dtype=np.int64
            )

        index_path = os.path.join(path, INDEX_FILE)
        with open(f"{index_path}.tmp", "wb") as f:
            np.savez(f, **arrays)
        os.replace(f"{index_path}.tmp", index_path)
        logger.debug("Index HNSW sauvegardé : %s", index_path)

    @classmethod
    def load(cls, path: str, matrix: np.ndarray) -> "HNSWIndex":
        """Recharge le graphe ; `matrix` est la matrice du store qui le porte."""
        with np.load(os.path.join(path, INDEX_FILE), allow_pickle=False) as data:
            m, ef_construction, ef_search, seed, entry_point = (int(v) for v in data["params"])
            index = cls(matrix.shape[1], m=m, ef_construction=ef_construction, ef_search=ef_search, seed=seed)
            index.levels = data["levels"].astype(np.int64)
            for layer in range(int(data["n_layers"])):
                nodes, sizes = data[f"nodes_{layer}"], data[f"sizes_{layer}"]
                neighbours = np.split(data[f"neighbours_{layer}"], np.cumsum(sizes)[:-1]) if len(sizes) else []
                index.links.append({int(n): ns.tolist() for n, ns in zip(nodes, neighbours, strict=True)})
        if len(index) > len(matrix):
            raise ValueError(f"HNSW index at {path} has {len(index)} nodes but the store only {len(matrix)} vectors")
        index.entry_point = None if entry_point < 0 else entry_point
        index._data = normalize(matrix[: len(index)]).astype(np.float32)
        # Nouvelle graine dérivée : les insertions suivantes restent reproductibles
        index._rng = np.random.default_rng([seed, len(index)])
        return index

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.isfile(os.path.join(path, INDEX_FILE))
