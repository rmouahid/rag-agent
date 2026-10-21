"""
Reranking des candidats par un cross-encoder, après la recherche vectorielle.

La recherche vectorielle compare deux embeddings calculés séparément : la
question et le chunk ne se « voient » jamais. Un cross-encoder lit la paire
(question, chunk) d'un seul tenant et note sa pertinence bien plus
finement, mais il coûte une passe du modèle par candidat : on ne l'applique
qu'aux `RERANK_CANDIDATES` premiers résultats de la recherche vectorielle.
"""

from typing import Protocol

import numpy as np

from rag_agent.config import config
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class PairScorer(Protocol):
    def predict(self, sentences: list[tuple[str, str]], batch_size: int = ...) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, model: PairScorer | None = None, batch_size: int = 16):
        """
        model_name : cross-encoder sentence-transformers (défaut :
        RERANKER_MODEL), téléchargé au premier lancement
        model      : modèle déjà chargé, à la place de model_name (tests)
        """
        self.model_name = model_name or config.reranker_model
        self.batch_size = batch_size
        if model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Chargement du cross-encoder : %s", self.model_name)
            model = CrossEncoder(self.model_name)
        self.model = model

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Chunks triés par pertinence selon le cross-encoder (`rerank_score`,
        le plus pertinent d'abord, égalités dans l'ordre d'origine). Le score
        cosine `score` est conservé : le filtre de pertinence s'appuie dessus.
        """
        if not chunks:
            return []
        scores = self.model.predict([(query, c["text"]) for c in chunks], batch_size=self.batch_size)
        reranked = [{**c, "rerank_score": float(s)} for c, s in zip(chunks, scores)]
        order = sorted(range(len(reranked)), key=lambda i: -reranked[i]["rerank_score"])
        return [reranked[i] for i in order]
