from fastapi import APIRouter, Depends, status

from rag_agent.api.dependencies import agent_lock, get_agent
from rag_agent.api.errors import APIError, error_responses
from rag_agent.api.schemas import AskRequest, AskResponse, SourceOut
from rag_agent.generation.answer import Answer
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.generation.truncation import ContextOverflowError
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["ask"])


def generation_failed() -> APIError:
    logger.exception("Échec de la génération de la réponse")
    return APIError(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "generation_failed",
        "The model failed to generate an answer.",
    )


def to_response(answer: Answer) -> AskResponse:
    return AskResponse(
        answer=answer.text.strip(),
        status=answer.status,
        sources=[SourceOut(**vars(s)) for s in answer.sources],
    )


@router.post("/ask", response_model=AskResponse, responses=error_responses(422, 500))
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
        raise APIError(status.HTTP_422_UNPROCESSABLE_ENTITY, "context_overflow", str(e))
    except Exception:
        raise generation_failed()
    return to_response(answer)
