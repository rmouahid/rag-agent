"""
Schémas Pydantic des requêtes et réponses de l'API. Ils valident les
entrées (FastAPI répond 422 sur une requête malformée) et décrivent chaque
route dans le schéma OpenAPI (`/docs`).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AskStatus = Literal["answered", "no_documents", "no_relevant_context"]


class Turn(BaseModel):
    """Un message d'un échange précédent."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"] = Field(description="Auteur du message")
    content: str = Field(max_length=8000, description="Texte du message")


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "Who built the Eiffel Tower?", "top_k": 5},
                {
                    "question": "When was it built?",
                    "history": [
                        {"role": "user", "content": "Who built the Eiffel Tower?"},
                        {"role": "assistant", "content": "Gustave Eiffel."},
                    ],
                },
            ]
        },
    )

    question: str = Field(min_length=1, max_length=4000, description="Question posée sur les documents indexés")
    top_k: int = Field(5, ge=1, le=50, description="Nombre de chunks récupérés avant le filtre de pertinence")
    history: list[Turn] = Field(
        default_factory=list,
        max_length=100,
        description="Échanges précédents, du plus ancien au plus récent ; l'API ne garde aucun état entre deux appels",
    )


class SourceOut(BaseModel):
    """Passage indexé placé dans le prompt pour construire la réponse."""

    source: str = Field(description="Fichier ou URL d'origine")
    excerpt: str = Field(description="Début du passage, espaces normalisés")
    chunk_id: str | None = Field(None, description="Identifiant du chunk dans le vector store")
    score: float | None = Field(None, description="Similarité cosinus avec la question")


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "Gustave Eiffel, for the 1889 World's Fair.",
                    "status": "answered",
                    "sources": [
                        {
                            "source": "france.md",
                            "excerpt": "The Eiffel Tower was built by...",
                            "chunk_id": "4b6d1eecef9805b6",
                            "score": 0.5,
                        }
                    ],
                }
            ]
        }
    )

    answer: str = Field(description="Réponse générée, ou message fixe si `status` n'est pas `answered`")
    status: AskStatus = Field(
        description=(
            "`answered`, ou `no_relevant_context` / `no_documents` quand aucun passage "
            "n'atteint `MIN_RELEVANCE_SCORE` ou que rien n'est indexé (le modèle n'est alors pas appelé)"
        )
    )
    sources: list[SourceOut] = Field(description="Passages utilisés, vide sans réponse générée")


class IngestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"source": "notes.md", "chunks_created": 2, "chunk_ids": ["96944874ef2c880e", "1f0c3a9b2d4e5f60"]}]
        }
    )

    source: str = Field(description="Nom du fichier reçu ou URL ingérée")
    chunks_created: int = Field(ge=0, description="Chunks nouvellement indexés ; 0 si la source l'était déjà")
    chunk_ids: list[str] = Field(description="Identifiants des chunks nouvellement indexés")


class HealthResponse(BaseModel):
    status: str = Field(description="`ok` si le service peut répondre")
    indexed_chunks: int = Field(ge=0, description="Nombre de chunks dans le vector store")
    model: str = Field(description="Chemin des poids GGUF chargés")
