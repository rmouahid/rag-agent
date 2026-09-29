from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from rag_agent.api.dependencies import agent_lock, get_agent
from rag_agent.generation.answer import Answer
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.generation.truncation import ContextOverflowError

router = APIRouter(tags=["ask"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, description="Question posée sur les documents indexés")
    top_k: int = Field(5, ge=1, le=50, description="Nombre de chunks récupérés avant le filtre de pertinence")
    history: list[Turn] = Field(
        default_factory=list,
        description="Échanges précédents, du plus ancien au plus récent ; l'API ne garde aucun état entre deux appels",
    )


class SourceOut(BaseModel):
    source: str
    excerpt: str
    chunk_id: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "no_documents", "no_relevant_context"]
    sources: list[SourceOut]


def to_response(answer: Answer) -> AskResponse:
    return AskResponse(
        answer=answer.text.strip(),
        status=answer.status,
        sources=[SourceOut(**vars(s)) for s in answer.sources],
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, agent: RAGAgent = Depends(get_agent)) -> AskResponse:
    """
    Répond à une question à partir des documents indexés, avec les sources
    utilisées. Si aucun passage n'est assez pertinent (`MIN_RELEVANCE_SCORE`)
    ou si rien n'est indexé, répond 200 avec un `status` explicite et sans
    générer de réponse, plutôt que de laisser le modèle inventer.
    """
    # Route synchrone : la génération (CPU) tourne dans un thread de FastAPI
    history = [turn.model_dump() for turn in request.history]
    try:
        with agent_lock:
            answer = agent.ask(request.question.strip(), top_k=request.top_k, history=history)
    except ContextOverflowError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return to_response(answer)
