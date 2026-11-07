from rag_agent.chunking.chunker import FixedSizeChunker
from rag_agent.chunking.sentence_chunker import SentenceChunker
from rag_agent.config import config


def get_chunker(
    strategy: str | None = None, chunk_size: int | None = None, overlap: int | None = None
) -> FixedSizeChunker | SentenceChunker:
    """
    Instancie le chunker correspondant à la stratégie demandée
    (par défaut : CHUNKING_STRATEGY via la config, #4).
    """
    strategy = strategy or config.chunking_strategy

    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        return SentenceChunker(chunk_size=chunk_size)

    raise ValueError(f"Unknown chunking strategy: {strategy!r} (expected 'fixed' or 'sentence')")
