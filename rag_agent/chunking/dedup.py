import re

from rag_agent.chunking.chunk import Chunk

_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_SIMILARITY_THRESHOLD = 0.9
_SHINGLE_SIZE = 5


class ShingleIndex:
    """
    Index inversé shingle → textes qui le contiennent, pour trouver les
    quasi-doublons sans comparer un chunk à tout le corpus.

    Deux textes de Jaccard ≥ seuil partagent forcément des shingles : seuls
    les textes qui partagent au moins un shingle avec le chunk sont
    candidats, et le Jaccard exact n'est calculé que pour eux. Le résultat
    est identique à une comparaison exhaustive, pour un coût proportionnel
    au nombre de candidats au lieu de la taille du corpus.

    Les textes déjà indexés (hash exact et shingles) sont conservés d'une
    ingestion à l'autre : `n_texts` permet de n'ajouter que les nouveaux.
    """

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.postings: dict[str, list[int]] = {}
        self.sizes: list[int] = []

    @property
    def n_texts(self) -> int:
        return len(self.sizes)

    def add_text(self, text: str) -> None:
        normalized = _normalize(text)
        self.hashes.add(_hash(normalized))
        self.add_shingles(_shingles(normalized))

    def add_shingles(self, shingles: set[str]) -> None:
        text_id = len(self.sizes)
        self.sizes.append(len(shingles))
        for shingle in shingles:
            self.postings.setdefault(shingle, []).append(text_id)

    def has_near_duplicate(self, shingles: set[str], threshold: float) -> bool:
        if not shingles:
            return False
        shared: dict[int, int] = {}
        for shingle in shingles:
            for text_id in self.postings.get(shingle, ()):
                shared[text_id] = shared.get(text_id, 0) + 1
        for text_id, intersection in shared.items():
            union = len(shingles) + self.sizes[text_id] - intersection
            if intersection / union >= threshold:
                return True
        return False


def deduplicate_chunks(
    chunks: list[Chunk],
    existing_texts: list[str] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    index: ShingleIndex | None = None,
) -> list[Chunk]:
    """
    Retire les chunks dupliqués ou quasi-dupliqués (typiquement issus
    de la ré-ingestion d'une même source, ou de deux sources qui se
    chevauchent) : d'abord un hash exact sur le texte normalisé (peu
    coûteux), puis une similarité de Jaccard sur des shingles de mots
    pour les quasi-doublons, limitée aux textes qui partagent au moins un
    shingle (ShingleIndex).

    Une comparaison par embeddings serait plus précise sur des
    reformulations, mais nettement plus coûteuse ; elle n'est pas
    nécessaire pour le cas visé ici (contenu répété ou fortement
    chevauchant) et resterait une option de repli si le
    hashing/shingling s'avérait insuffisant en pratique.

    `existing_texts` permet de dédupliquer aussi contre un corpus déjà
    indexé (ré-ingestion), pas seulement au sein du batch courant.
    `index` évite de ré-indexer ce corpus à chaque appel : les textes
    d'`existing_texts` au-delà de `index.n_texts` y sont ajoutés, et les
    chunks conservés aussi, si bien qu'il reste à jour pour l'appel suivant
    pourvu que ces chunks soient ensuite ajoutés au corpus.
    """
    existing_texts = existing_texts or []
    if index is None or index.n_texts > len(existing_texts):
        index = ShingleIndex()
    for text in existing_texts[index.n_texts :]:
        index.add_text(text)

    kept: list[Chunk] = []
    for chunk in chunks:
        normalized = _normalize(chunk.text)
        exact_hash = _hash(normalized)
        if exact_hash in index.hashes:
            continue

        shingles = _shingles(normalized)
        if index.has_near_duplicate(shingles, similarity_threshold):
            continue

        index.hashes.add(exact_hash)
        index.add_shingles(shingles)
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
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0
