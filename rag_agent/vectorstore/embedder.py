from sentence_transformers import SentenceTransformer
import numpy as np

from rag_agent.config import config
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(self, model_name: str = None):
        """
        model_name : modèle sentence-transformers à charger.
        Par défaut, lu depuis la config (EMBEDDING_MODEL).
        Se télécharge automatiquement au premier lancement.
        """
        model_name = model_name or config.embedding_model
        logger.info("Chargement du modèle d'embedding : %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Modèle chargé !")

    def embed(self, text: str) -> np.ndarray:
        """
        Transforme un texte en vecteur numpy
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """
        Transforme une liste de textes en matrice de vecteurs
        """
        return self.model.encode(texts, convert_to_numpy=True)
