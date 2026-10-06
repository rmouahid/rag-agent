"""
Gestion centralisée des erreurs : toute réponse d'erreur de l'API a la
même forme, `{"error": <code>, "detail": <explication>}`, et aucune trace
d'exécution n'est renvoyée au client (elle reste dans les logs).
"""

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    error: str = Field(description="Code stable de l'erreur, en snake_case (ex. `unsupported_source`)")
    detail: Any = Field(description="Explication lisible ; pour `validation_error`, la liste des champs refusés")


class APIError(Exception):
    """
    Erreur prévue, levée par une route : convertie telle quelle en réponse
    `status_code` avec l'enveloppe d'erreur
    """

    def __init__(self, status_code: int, error: str, detail: Any):
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail


def error_responses(*codes: int) -> dict:
    """
    Réponses d'erreur d'une route, pour le schéma OpenAPI
    """
    return {code: {"model": ErrorResponse, "description": HTTPStatus(code).phrase} for code in codes}


def _envelope(status_code: int, error: str, detail: Any, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail}, headers=headers)


def _code_for(status_code: int) -> str:
    """
    404 → "not_found", 405 → "method_not_allowed"...
    """
    try:
        return HTTPStatus(status_code).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "http_error"


async def _api_error(request: Request, exc: APIError) -> JSONResponse:
    return _envelope(exc.status_code, exc.error, exc.detail)


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _envelope(exc.status_code, _code_for(exc.status_code), exc.detail, getattr(exc, "headers", None))


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Sans l'entrée reçue (`input`), qui peut être volumineuse ou sensible
    fields = [
        {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"], "type": err["type"]}
        for err in exc.errors()
    ]
    return _envelope(status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", fields)


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erreur inattendue sur %s %s", request.method, request.url.path)
    return _envelope(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred. See the server logs for details.",
    )


def install_error_handlers(app: FastAPI) -> None:
    # Starlette type tout handler comme prenant une `Exception`, alors qu'il
    # n'appelle chacun qu'avec la classe sous laquelle il est enregistré :
    # un handler qui annonce sa sous-classe est correct, mais refusé par mypy
    app.add_exception_handler(APIError, _api_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unexpected_error)
