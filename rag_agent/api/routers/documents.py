import os
import shutil
import tempfile
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from rag_agent.api.dependencies import agent_lock, get_agent
from rag_agent.generation.rag_agent import RAGAgent, UnsupportedSourceError, source_type
from rag_agent.ingestion.url_guard import BlockedURLError

router = APIRouter(tags=["documents"])

class IngestResponse(BaseModel):
    source: str
    chunks_created: int
    chunk_ids: list[str]


def _unsupported(error: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error))


def _ingest_upload(agent: RAGAgent, file: UploadFile) -> list[str]:
    """
    Les loaders lisent un chemin : le fichier reçu est copié dans un
    fichier temporaire de même extension, puis supprimé
    """
    name = os.path.basename(file.filename or "")
    try:
        source_type(name)
    except UnsupportedSourceError as e:
        raise _unsupported(e)

    suffix = os.path.splitext(name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        with agent_lock:
            return agent.ingest(tmp.name, source_name=name)
    finally:
        os.unlink(tmp.name)


def _ingest_url(agent: RAGAgent, url: str) -> list[str]:
    try:
        if source_type(url) != "web":
            raise UnsupportedSourceError(f"Unsupported URL {url!r}: expected an http(s) URL")
        with agent_lock:
            return agent.ingest(url)
    except UnsupportedSourceError as e:
        raise _unsupported(e)
    except BlockedURLError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except ValueError as e:
        # Page atteinte mais contenu non HTML
        raise _unsupported(e)


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
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
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Send exactly one of 'file' or 'url'.",
        )

    if file is not None:
        source = os.path.basename(file.filename or "")
        chunk_ids = _ingest_upload(agent, file)
    else:
        source = url
        chunk_ids = _ingest_url(agent, url)

    return IngestResponse(source=source, chunks_created=len(chunk_ids), chunk_ids=chunk_ids)
