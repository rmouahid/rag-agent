import numpy as np


def cosine_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Recherche exacte des k lignes de `matrix` les plus proches de `query`
    en similarité cosine (scan complet, numpy uniquement).

    C'est la baseline de correction : les index approximatifs (IVF, ...)
    sont évalués par leur recall face à ce résultat, donc il privilégie
    l'exactitude (calcul en float64, aucun epsilon qui fausse les scores)
    à la vitesse.

    Retourne (indices, scores), triés par score décroissant. Les égalités
    sont départagées par l'indice le plus petit, pour un résultat
    déterministe. Un vecteur nul (ligne ou requête) a un score de 0.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if query.shape != (matrix.shape[1],):
        raise ValueError(f"query shape {query.shape} does not match vector dimension {matrix.shape[1]}")

    # Cosine = produit scalaire des vecteurs normalisés ; norme nulle -> score 0
    row_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    denominator = row_norms * query_norm
    scores = np.zeros(len(matrix))
    np.divide(matrix @ query, denominator, out=scores, where=denominator > 0)

    n = len(scores)
    k = min(k, n)
    if k == n:
        candidates = np.arange(n)
    else:
        # Seuil du k-ième meilleur score, puis tous les ex æquo du seuil : le
        # départage par indice reste exact même quand l'égalité coupe le top-k
        threshold = np.partition(scores, n - k)[n - k]
        candidates = np.flatnonzero(scores >= threshold)

    order = np.lexsort((candidates, -scores[candidates]))[:k]
    top = candidates[order]
    return top, scores[top]
