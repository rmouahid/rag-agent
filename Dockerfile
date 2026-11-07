# RAG Agent API image.
#
#   docker build -t rag-agent .
#   docker run -p 8000:8000 \
#       -v "$PWD/models:/app/models:ro" \
#       -v rag-agent-index:/app/vector_db \
#       -v rag-agent-cache:/app/.cache \
#       rag-agent
#
# Model weights, the vector store and the caches (embedding model, computed
# embeddings) are volumes, never baked into the image: the image stays small
# and the index survives container upgrades. See docker-compose.yml.

# --- Build stage: compile llama-cpp-python, install the dependencies --------
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    # Portable binary: no -march=native, so the image runs on any x86-64 CPU
    CMAKE_ARGS="-DGGML_NATIVE=OFF"

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
# Dependencies first, in their own layer: code and README changes do not
# recompile llama.cpp. Poetry only needs the declared README to exist here
RUN touch README.md && poetry install --only main --no-root
COPY README.md ./
COPY rag_agent ./rag_agent
RUN poetry install --only main

# --- Runtime stage -----------------------------------------------------------
FROM python:3.12-slim

# libgomp: OpenMP runtime used by llama.cpp and torch
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 rag

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/rag_agent /app/rag_agent

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    # The sentence-transformers model is downloaded once into the cache volume
    HF_HOME=/app/.cache/huggingface \
    LOCAL_MODEL_PATH=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    VECTOR_STORE_PATH=vector_db \
    EMBEDDING_CACHE_DIR=.cache/embeddings

RUN mkdir -p models vector_db .cache && chown -R rag:rag /app/vector_db /app/.cache
VOLUME ["/app/vector_db", "/app/.cache"]
USER rag

EXPOSE 8000

# /health answers 503 until the model and the store are ready; loading the
# weights takes a while, hence the start period
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

CMD ["uvicorn", "rag_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
