from dataclasses import dataclass, field


@dataclass
class Document:
    """
    Représentation normalisée d'une source ingérée, partagée par tous
    les loaders (PDF, Markdown, web).
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)
