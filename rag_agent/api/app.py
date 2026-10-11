"""
Point d'entrée du service HTTP :

    poetry run uvicorn rag_agent.api.app:app

Le RAGAgent (poids du LLM, vector store) est construit une seule fois au
démarrage et partagé par toutes les requêtes.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_agent.api.errors import install_error_handlers
from rag_agent.api.limits import MaxBodySizeMiddleware
from rag_agent.api.routers import ask, documents, health
from rag_agent.config import config
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


def create_app(agent_factory: Callable[[], RAGAgent] = RAGAgent, max_body_bytes: int | None = None) -> FastAPI:
    """
    Construit l'application ; `agent_factory` n'est appelée qu'au démarrage,
    ce qui permet aux tests d'en fournir une sans charger de modèle.
    `max_body_bytes` borne la taille des requêtes (défaut : MAX_UPLOAD_MB)
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    install_error_handlers(app)
    app.state.max_body_bytes = config.max_upload_mb * 1024 * 1024 if max_body_bytes is None else max_body_bytes
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=app.state.max_body_bytes)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    return app


app = create_app()
