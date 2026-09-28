"""
Point d'entrée du service HTTP :

    poetry run uvicorn rag_agent.api.app:app

Le RAGAgent (poids du LLM, vector store) est construit une seule fois au
démarrage et partagé par toutes les requêtes.
"""

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from rag_agent.api.routers import documents, health
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


def create_app(agent_factory: Callable[[], RAGAgent] = RAGAgent) -> FastAPI:
    """
    Construit l'application ; `agent_factory` n'est appelée qu'au démarrage,
    ce qui permet aux tests d'en fournir une sans charger de modèle
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Sans poids, ModelNotFoundError interrompt le démarrage avec la
        # commande de téléchargement, comme en CLI
        logger.info("Initialisation du RAG agent...")
        app.state.agent = agent_factory()
        yield

    app = FastAPI(
        title="RAG Agent",
        description="Retrieval-Augmented Generation over your documents, with a local LLM.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    return app


app = create_app()
