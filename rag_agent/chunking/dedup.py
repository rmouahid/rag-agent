import re

from rag_agent.chunking.chunk import Chunk

_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_SIMILARITY_THRESHOLD = 0.9
_SHINGLE_SIZE = 5


def deduplicate_chunks(
    chunks: list[Chunk],
    existing_texts: list[str] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[Chunk]:
    """
    Retire les chunks dupliqués ou quasi-dupliqués (typiquement issus
    de la ré-ingestion d'une même source, ou de deux sources qui se
    chevauchent) : d'abord un hash exact sur le texte normalisé (peu
    coûteux), puis une similarité de Jaccard sur des shingles de mots
    pour les quasi-doublons.

    Une comparaison par embeddings serait plus précise sur des
    reformulations, mais nettement plus coûteuse ; elle n'est pas
    nécessaire pour le cas visé ici (contenu répété ou fortement
    chevauchant) et resterait une option de repli si le
    hashing/shingling s'avérait insuffisant en pratique.

    `existing_texts` permet de dédupliquer aussi contre un corpus déjà
    indexé (ré-ingestion), pas seulement au sein du batch courant.
    """
    existing_texts = existing_texts or []

    seen_hashes = {_hash(_normalize(t)) for t in existing_texts}
    kept_shingles = [_shingles(_normalize(t)) for t in existing_texts]
    kept: list[Chunk] = []

    for chunk in chunks:
        normalized = _normalize(chunk.text)
        exact_hash = _hash(normalized)
        if exact_hash in seen_hashes:
            continue

        shingles = _shingles(normalized)
        if any(_jaccard(shingles, other) >= similarity_threshold for other in kept_shingles):
            continue

        seen_hashes.add(exact_hash)
        kept_shingles.append(shingles)
        kept.append(chunk)

    return kept


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0
