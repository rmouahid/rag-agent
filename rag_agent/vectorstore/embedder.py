import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_agent.config import config
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """
    Cache d'embeddings indexé par `Chunk.id`.

    L'id d'un chunk est un hash de (source, position, texte) : tant que
    le chunk n'a pas changé, son embedding reste valide. Le cache est
    persisté dans un fichier .npz (ids + matrice) pour survivre aux
    relances ; sans `path`, il reste purement en mémoire.
    """

    def __init__(self, path: str = None):
        self.path = path
        self._vectors: dict[str, np.ndarray] = {}
        if path and os.path.exists(path):
            self._load()

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def get(self, chunk_id: str) -> np.ndarray:
        return self._vectors[chunk_id]

    def put(self, chunk_id: str, vector: np.ndarray) -> None:
        self._vectors[chunk_id] = vector

    def save(self) -> None:
        if not self.path or not self._vectors:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        ids = np.array(list(self._vectors), dtype=str)
        vectors = np.stack(list(self._vectors.values()))
        # Écriture atomique : un crash en cours de save ne corrompt pas le cache
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "wb") as f:
            np.savez(f, ids=ids, vectors=vectors)
        os.replace(tmp_path, self.path)
        logger.debug("Cache d'embeddings sauvegardé : %s (%d entrées)", self.path, len(ids))

    def _load(self) -> None:
        try:
            with np.load(self.path, allow_pickle=False) as data:
                self._vectors = dict(zip(data["ids"].tolist(), data["vectors"]))
        except Exception as e:  # fichier tronqué / illisible : le cache se reconstruit
            logger.warning("Cache d'embeddings illisible (%s), il sera reconstruit : %s", self.path, e)
            self._vectors = {}
            return
        logger.info("Cache d'embeddings chargé : %s (%d entrées)", self.path, len(self._vectors))


class Embedder:
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        model_name : modèle sentence-transformers à charger.
        Par défaut, lu depuis la config (EMBEDDING_MODEL).
        Se télécharge automatiquement au premier lancement.

        cache_dir : dossier du cache d'embeddings (un fichier par modèle,
        car les vecteurs de deux modèles ne sont pas interchangeables).
        Par défaut, lu depuis la config (EMBEDDING_CACHE_DIR) ; une chaîne
        vide désactive la persistance (cache en mémoire uniquement).
        """
        model_name = model_name or config.embedding_model
        cache_dir = config.embedding_cache_dir if cache_dir is None else cache_dir

        logger.info("Chargement du modèle d'embedding : %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info("Modèle chargé !")

        cache_path = None
        if cache_dir:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
            cache_path = os.path.join(cache_dir, f"{slug}.npz")
        self.cache = EmbeddingCache(cache_path)

    def embed(self, text: str) -> np.ndarray:
        """
        Transforme un texte en vecteur numpy
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Transforme une liste de textes en matrice de vecteurs
        """
        return self.model.encode(texts, convert_to_numpy=True, batch_size=batch_size)

    def embed_chunks(self, chunks: list, batch_size: int = 64) -> np.ndarray:
        """
        Embed un batch de Chunk, en ne calculant que ceux absents du cache.

        Retourne une matrice (len(chunks) x dim), dans l'ordre des chunks.
        Les chunks partageant un même id ne sont encodés qu'une fois.
        """
        if not chunks:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        missing = {}
        for chunk in chunks:
            if chunk.id not in self.cache:
                missing.setdefault(chunk.id, chunk.text)

        if missing:
            logger.debug("%d/%d chunks à encoder (le reste vient du cache)", len(missing), len(chunks))
            vectors = self.embed_many(list(missing.values()), batch_size=batch_size)
            for chunk_id, vector in zip(missing, vectors):
                self.cache.put(chunk_id, vector)
            self.cache.save()
        else:
            logger.debug("%d chunks servis depuis le cache", len(chunks))

        return np.stack([self.cache.get(chunk.id) for chunk in chunks])
