import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", "")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "fixed")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vector_db")
    index_type: str = os.getenv("INDEX_TYPE", "brute-force")
    nprobe: int = int(os.getenv("NPROBE", "8"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
