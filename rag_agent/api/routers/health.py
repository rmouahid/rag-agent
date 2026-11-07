from fastapi import APIRouter, Depends, Response, status

from rag_agent.api.dependencies import get_llm, get_vector_store
from rag_agent.api.errors import error_responses
from rag_agent.api.schemas import HealthChecks, HealthResponse
from rag_agent.generation.llm import LocalLLM
from rag_agent.vectorstore.vector_store import VectorStore

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "A component is not ready (see `checks`)",
        },
        **error_responses(500),
    },
)
def health(
    response: Response,
    vector_store: VectorStore = Depends(get_vector_store),
    llm: LocalLLM = Depends(get_llm),
) -> HealthResponse:
    """
    Le modèle est-il chargé, le vector store prêt ? 200 si oui, 503 sinon :
    utilisable tel quel comme healthcheck Docker
    """
    checks = HealthChecks(llm=llm.is_loaded, vector_store=vector_store.is_ready)
    ready = checks.llm and checks.vector_store
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else "unavailable",
        checks=checks,
        indexed_chunks=len(vector_store.chunks),
        model=llm.model_path,
    )
