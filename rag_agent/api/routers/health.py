from fastapi import APIRouter, Depends

from rag_agent.api.dependencies import get_llm, get_vector_store
from rag_agent.api.schemas import HealthResponse
from rag_agent.generation.llm import LocalLLM
from rag_agent.vectorstore.vector_store import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    vector_store: VectorStore = Depends(get_vector_store),
    llm: LocalLLM = Depends(get_llm),
) -> HealthResponse:
    """
    Le service répond, le modèle est chargé ; nombre de chunks indexés
    """
    return HealthResponse(status="ok", indexed_chunks=len(vector_store.chunks), model=llm.model_path)
