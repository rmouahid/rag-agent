import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

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
        raise APIError(status.HTTP_422_UNPROCESSABLE_ENTITY, "context_overflow", str(e)) from e
    except Exception as e:
        raise generation_failed() from e
    return to_response(answer)


# Réponse en flux : Server-Sent Events
STREAM_RESPONSES = {
    status.HTTP_200_OK: {
        "description": (
            "Flux `text/event-stream` : un événement `sources` (`status` et `sources`, "
            'comme `/ask`), puis un `token` par morceau de texte (`{"text": ...}`), '
            'puis `done` (`{"answer": <texte complet>}`). Une erreur pendant la '
            'génération arrive comme un événement `error` (`{"error", "detail"}`) '
            "qui termine le flux."
        ),
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    },
    **error_responses(422),
}
# Pas de mise en tampon par un proxy (nginx) ni de cache : chaque token part tout de suite
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse(event: str, data: dict) -> str:
    """
    Un événement SSE ; le JSON tient sur une ligne, sauts de ligne du texte compris
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error(error: str, detail: str) -> str:
    return sse("error", {"error": error, "detail": detail})


def answer_events(agent: RAGAgent, question: str, top_k: int, history: list[dict]) -> Iterator[str]:
    """
    Événements SSE d'une réponse. Le verrou de l'agent est pris par le
    générateur lui-même, pour toute la génération, et relâché dès que le
    flux se termine ou que le client se déconnecte (fermeture du générateur).
    Le statut HTTP (200) étant déjà parti, les erreurs deviennent des
    événements `error`.
    """
    with agent_lock:
        try:
            answer = agent.ask_stream(question, top_k=top_k, history=history)
        except ContextOverflowError as e:
            yield _stream_error("context_overflow", str(e))
            return
        except Exception:
            logger.exception("Échec de la préparation de la réponse en flux")
            yield _stream_error("generation_failed", "The model failed to generate an answer.")
            return

        yield sse(
            "sources",
            {"status": answer.status, "sources": [SourceOut(**vars(s)).model_dump() for s in answer.sources]},
        )

        pieces: list[str] = []
        try:
            for piece in answer:
                if not pieces:
                    # Comme `/ask` : pas d'espaces en tête de réponse
                    piece = piece.lstrip()
                if piece:
                    pieces.append(piece)
                    yield sse("token", {"text": piece})
        except Exception:
            logger.exception("Échec de la génération en flux")
            yield _stream_error("generation_failed", "The model failed to generate an answer.")
            return

        yield sse("done", {"answer": "".join(pieces).strip()})


def _event_stream(events: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream", headers=STREAM_HEADERS)


@router.post("/ask/stream", response_class=StreamingResponse, responses=STREAM_RESPONSES)
def ask_stream(request: AskRequest, agent: RAGAgent = Depends(get_agent)) -> StreamingResponse:
    """
    Comme `/ask`, mais la réponse arrive token par token, en Server-Sent
    Events, au fil de la génération. À lire avec `fetch` (corps JSON,
    `history` possible) ou `curl -N`.
    """
    history = [turn.model_dump() for turn in request.history]
    return _event_stream(answer_events(agent, request.question.strip(), request.top_k, history))


@router.get("/ask/stream", response_class=StreamingResponse, responses=STREAM_RESPONSES)
def ask_stream_get(
    question: str = Query(min_length=1, max_length=4000, description="Question posée sur les documents indexés"),
    top_k: int = Query(5, ge=1, le=50, description="Nombre de chunks récupérés avant le filtre de pertinence"),
    agent: RAGAgent = Depends(get_agent),
) -> StreamingResponse:
    """
    Variante GET de `POST /ask/stream`, pour `EventSource` dans un
    navigateur (qui ne sait faire que des GET) : question en paramètre de
    requête, sans historique.
    """
    return _event_stream(answer_events(agent, question.strip(), top_k, []))
