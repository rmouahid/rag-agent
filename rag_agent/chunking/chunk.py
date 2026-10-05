import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    Unité indexable partagée entre le chunking, le vector store et la
    génération. `id` est un hash déterministe de (source, position,
    text) — deux ingestions du même passage produisent le même id,
    ce que la déduplication (#14) exploite.
    """

    text: str
    source: str
    position: int
    metadata: dict = field(default_factory=dict)
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            self.id = self._compute_id()

    def _compute_id(self) -> str:
        digest = hashlib.sha256(f"{self.source}:{self.position}:{self.text}".encode())
        return digest.hexdigest()[:16]
