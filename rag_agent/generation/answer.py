from dataclasses import dataclass, field
from typing import Iterator

EXCERPT_LENGTH = 200


def _excerpt(text: str, length: int = EXCERPT_LENGTH) -> str:
    """
    Début du chunk, espaces normalisés, coupé sur une fin de mot
    """
    text = " ".join(text.split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "…"


@dataclass(frozen=True)
class Source:
    """
    Chunk effectivement placé dans le prompt : ce qui permet de vérifier
    sur quoi la réponse s'appuie
    """

    source: str
    excerpt: str
    chunk_id: str | None = None
    score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: dict) -> "Source":
        return cls(
            source=chunk["source"],
            excerpt=_excerpt(chunk["text"]),
            chunk_id=chunk.get("id"),
            score=chunk.get("score"),
        )


@dataclass(frozen=True)
class Answer:
    """
    Réponse générée et sources utilisées pour la construire
    """

    text: str
    sources: list[Source] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text

    def format_sources(self) -> str:
        """
        Liste numérotée des sources, une par ligne ; vide sans source
        """
        return "\n".join(
            f"[{i}] {s.source} — {s.excerpt}" for i, s in enumerate(self.sources, start=1)
        )


@dataclass(frozen=True)
class StreamingAnswer:
    """
    Réponse en cours de génération : les sources sont connues avant le
    premier token (le prompt est déjà construit), le texte arrive par
    `tokens`, à consommer une seule fois
    """

    tokens: Iterator[str]
    sources: list[Source] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def format_sources(self) -> str:
        return Answer(text="", sources=self.sources).format_sources()
