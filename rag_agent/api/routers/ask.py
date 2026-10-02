from fastapi import APIRouter, Depends, HTTPException, status

from rag_agent.api.dependencies import agent_lock, get_agent
from rag_agent.api.schemas import AskRequest, AskResponse, SourceOut
from rag_agent.generation.answer import Answer
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.generation.truncation import ContextOverflowError

router = APIRouter(tags=["ask"])


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
