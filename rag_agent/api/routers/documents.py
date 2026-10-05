import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pymupdf import FileDataError

from rag_agent.api.dependencies import agent_lock, get_agent
from rag_agent.api.errors import APIError, error_responses
from rag_agent.api.schemas import IngestResponse
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.ingestion.url_guard import BlockedURLError
from rag_agent.logging_config import get_logger
from rag_agent.pipeline import UnsupportedSourceError, source_type

logger = get_logger(__name__)

router = APIRouter(tags=["documents"])


def _unsupported(error: Exception) -> APIError:
    return APIError(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported_source", str(error))


def _ingestion_failed(source: str) -> APIError:
    # Cause dans les logs : le message brut peut contenir des chemins du serveur
    logger.exception("Échec de l'ingestion de %s", source)
    return APIError(status.HTTP_500_INTERNAL_SERVER_ERROR, "ingestion_failed", f"Could not ingest {source!r}.")


def _ingest_upload(agent: RAGAgent, file: UploadFile) -> list[str]:
    """
    Les loaders lisent un chemin : le fichier reçu est copié dans un
    fichier temporaire de même extension, puis supprimé
    """
    name = os.path.basename(file.filename or "")
    try:
        source_type(name)
    except UnsupportedSourceError as e:
        raise _unsupported(e) from e

    suffix = os.path.splitext(name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        with agent_lock:
            return agent.ingest(tmp.name, source_name=name)
    except (UnicodeDecodeError, FileDataError) as e:
        # Fichier illisible : PDF corrompu (PyMuPDF), Markdown non UTF-8...
        logger.info("Fichier illisible %s : %s", name, e)
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "unreadable_file",
            f"Could not read {name!r}: the file is corrupted or not a valid {suffix} file.",
        ) from e
    except Exception as e:
        raise _ingestion_failed(name) from e
    finally:
        os.unlink(tmp.name)


def _ingest_url(agent: RAGAgent, url: str) -> list[str]:
    try:
        if source_type(url) != "web":
            raise UnsupportedSourceError(f"Unsupported URL {url!r}: expected an http(s) URL")
        with agent_lock:
            return agent.ingest(url)
    except UnsupportedSourceError as e:
        raise _unsupported(e) from e
    except BlockedURLError as e:
        raise APIError(status.HTTP_400_BAD_REQUEST, "blocked_url", str(e)) from e
    except ConnectionError as e:
        raise APIError(status.HTTP_502_BAD_GATEWAY, "source_unreachable", str(e)) from e
    except ValueError as e:
        # Page atteinte mais contenu non HTML
        raise _unsupported(e) from e
    except Exception as e:
        raise _ingestion_failed(url) from e


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(400, 415, 422, 500, 502),
)
def ingest_document(
    file: UploadFile | None = File(None, description="PDF or Markdown file"),
    url: str | None = Form(None, description="http(s) URL of an HTML page"),
    agent: RAGAgent = Depends(get_agent),
) -> IngestResponse:
    """
    Ingère un fichier (PDF, Markdown) ou une page web : chunking,
    embeddings et indexation. Les passages déjà indexés sont ignorés, donc
    `chunks_created` peut valoir 0 pour une source déjà ingérée.
    """
    # Route synchrone : FastAPI l'exécute dans un thread, l'ingestion
    # (embeddings sur CPU) ne bloque pas la boucle d'événements
    if (file is None) == (url is None):
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_source",
            "Send exactly one of 'file' or 'url'.",
        )

    if file is not None:
        source = os.path.basename(file.filename or "")
        chunk_ids = _ingest_upload(agent, file)
    else:
        source = url
        chunk_ids = _ingest_url(agent, url)

    return IngestResponse(source=source, chunks_created=len(chunk_ids), chunk_ids=chunk_ids)
