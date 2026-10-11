# syntax=docker/dockerfile:1

# ---------- build: dependencies in a virtualenv --------------------------------------------
FROM python:3.12.13-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src
# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------- models: embedding model baked into the image -----------------------------------
# The first start then needs no network, and indexing / search work without any LLM.
# Set PRELOAD_EMBEDDINGS=0 when using EMBEDDING_PROVIDER=ollama|openai|chroma.
FROM build AS models
ARG PRELOAD_EMBEDDINGS=1
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large
RUN mkdir -p /models/fastembed && \
    if [ "$PRELOAD_EMBEDDINGS" = "1" ]; then \
      /opt/venv/bin/python -c "from fastembed import TextEmbedding; \
TextEmbedding(model_name='${EMBEDDING_MODEL}', cache_dir='/models/fastembed')"; \
    fi

# ---------- runtime ------------------------------------------------------------------------
FROM python:3.12.13-slim-bookworm AS runtime

RUN useradd --create-home --uid 1000 app && \
    mkdir -p /data/chroma /app/sample_docs /app/docs && \
    chown -R app:app /data /app

COPY --from=build /opt/venv /opt/venv
COPY --from=models --chown=app:app /models /models

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    CHROMA_PATH=/data/chroma \
    EMBEDDING_CACHE_DIR=/models/fastembed \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Relative paths given to index_folder (e.g. "./sample_docs") resolve against /app.
WORKDIR /app
USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('SERVER_PORT', '8000') + '/health', timeout=4)"

CMD ["rag-kb"]
