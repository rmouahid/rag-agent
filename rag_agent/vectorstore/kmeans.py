from dataclasses import dataclass

import numpy as np

from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

INIT_METHODS = ("k-means++", "random")


@dataclass
class KMeansResult:
    """
    Résultat d'un k-means :
    - centroids : matrice (n_clusters x dim) des centres
    - labels    : cluster assigné à chaque vecteur (N,)
    - inertia   : somme des distances² de chaque vecteur à son centre
    - n_iter    : nombre d'itérations de Lloyd effectuées
    - converged : True si l'arrêt vient du seuil `tol` (et non de `max_iter`)
    """

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    converged: bool


def squared_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Distances euclidiennes² entre chaque vecteur et chaque centre (N x K),
    via ||x||² - 2 x·c + ||c||² : un seul produit matriciel, sans tenseur
    intermédiaire N x K x dim. Les petites valeurs négatives dues aux
    arrondis sont ramenées à 0.
    """
    distances = (
        np.einsum("ij,ij->i", vectors, vectors)[:, None]
        - 2.0 * (vectors @ centroids.T)
        + np.einsum("ij,ij->i", centroids, centroids)[None, :]
    )
    return np.maximum(distances, 0.0, out=distances)


def assign(vectors: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigne chaque vecteur à son centre le plus proche.
    Retourne (labels, distances² au centre retenu).
    """
    distances = squared_distances(vectors, centroids)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(vectors)), labels]


def _init_random(vectors: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    return vectors[rng.choice(len(vectors), size=n_clusters, replace=False)].copy()


def _init_kmeans_plus_plus(vectors: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ (Arthur & Vassilvitskii, 2007) : chaque nouveau centre est
    tiré avec une probabilité proportionnelle à la distance² au centre
    déjà choisi le plus proche, ce qui étale les centres initiaux.
    """
    centroids = np.empty((n_clusters, vectors.shape[1]), dtype=vectors.dtype)
    centroids[0] = vectors[rng.integers(len(vectors))]
    closest = squared_distances(vectors, centroids[:1])[:, 0]

    for i in range(1, n_clusters):
        total = closest.sum()
        if total > 0:
            idx = rng.choice(len(vectors), p=closest / total)
        else:  # tous les points coïncident avec un centre : tirage uniforme
            idx = rng.integers(len(vectors))
        centroids[i] = vectors[idx]
        closest = np.minimum(closest, squared_distances(vectors, centroids[i : i + 1])[:, 0])

    return centroids


def kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    init: str = "k-means++",
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = None,
) -> KMeansResult:
    """
    k-means de Lloyd, implémenté avec numpy uniquement (quantificateur
    grossier de l'index IVF).

    init     : "k-means++" (défaut) ou "random" (n_clusters points tirés
               sans remise)
    max_iter : nombre maximal d'itérations de Lloyd
    tol      : arrêt dès que le plus grand déplacement d'un centre entre
               deux itérations passe sous `tol` × (écart-type moyen des
               données), un seuil relatif qui ne dépend pas de l'échelle
    seed     : graine du générateur aléatoire, pour un résultat reproductible

    Un cluster qui se vide est ré-amorcé sur le point le plus éloigné de
    son centre actuel, pour toujours retourner exactement n_clusters centres.
    """
    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if n_clusters > len(vectors):
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed the number of vectors ({len(vectors)})")
    if init not in INIT_METHODS:
        raise ValueError(f"init must be one of {INIT_METHODS}, got {init!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    rng = np.random.default_rng(seed)
    if init == "k-means++":
        centroids = _init_kmeans_plus_plus(vectors, n_clusters, rng)
    else:
        centroids = _init_random(vectors, n_clusters, rng)

    threshold = tol * float(np.mean(np.std(vectors, axis=0)))
    converged = False

    for n_iter in range(1, max_iter + 1):  # noqa: B007 (read after the loop)
        labels, distances = assign(vectors, centroids)

        # Étape de mise à jour : moyenne des points de chaque cluster
        counts = np.bincount(labels, minlength=n_clusters)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, vectors)
        new_centroids = centroids.copy()
        filled = counts > 0
        new_centroids[filled] = sums[filled] / counts[filled, None]

        # Clusters vides : ré-amorcés sur les points les plus mal représentés
        empty = np.flatnonzero(~filled)
        if len(empty):
            farthest = np.argsort(-distances, kind="stable")[: len(empty)]
            new_centroids[empty] = vectors[farthest]
            logger.debug("k-means : %d cluster(s) vide(s) ré-amorcé(s)", len(empty))

        shift = float(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max())
        centroids = new_centroids
        if shift <= threshold and not len(empty):
            converged = True
            break

    labels, distances = assign(vectors, centroids)
    logger.debug("k-means : %d clusters, %d itérations, convergé=%s", n_clusters, n_iter, converged)
    return KMeansResult(
        centroids=centroids,
        labels=labels,
        inertia=float(distances.sum()),
        n_iter=n_iter,
        converged=converged,
    )
