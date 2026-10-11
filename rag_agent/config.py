import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llm_context_size: int = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_context_margin: int = int(os.getenv("LLM_CONTEXT_MARGIN", "64"))
    llm_threads: int = int(os.getenv("LLM_THREADS", "0"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "fixed")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vector_db")
    index_type: str = os.getenv("INDEX_TYPE", "brute-force")
    nprobe: int = int(os.getenv("NPROBE", "8"))
    ivf_n_lists: int = int(os.getenv("IVF_NLIST", "0"))
    ivf_min_vectors: int = int(os.getenv("IVF_MIN_VECTORS", "10000"))
    ivf_rebuild_threshold: float = float(os.getenv("IVF_REBUILD_THRESHOLD", "0.2"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    allow_private_urls: bool = os.getenv("ALLOW_PRIVATE_URLS", "false").lower() in ("1", "true", "yes")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
