# Runtime image for the Streamlit app (Stage 5). Ingestion runs offline, separately,
# against the same managed Postgres — it is NOT part of this image's steady state
# (see docs/architecture.md §3.13).
FROM python:3.13-slim

# curl: container healthcheck only. psycopg[binary] ships its own libpq, so no
# build toolchain or libpq-dev is needed here.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# `-e` (editable): config.py locates config/corpora/*.yaml by walking up from its
# own file, which only resolves correctly when the package lives inside the repo
# tree — same as local dev. A regular install copies docqa/ into site-packages and
# that resolution breaks (REPO_ROOT lands in /usr/local/lib/...). `agents`
# (langgraph, google-genai) + `app` (streamlit); `dev`/`eval` extras are dev-only.
RUN pip install --no-cache-dir -e ".[agents,app]"

# Bake the embedding + reranker weights into the image at build time: the running
# container never depends on Hugging Face Hub being reachable, and there's no
# multi-hundred-MB download stalling the first request after a cold start.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

ENV PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

# Streamlit's own health endpoint — used by both fly.toml and railway.json.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/_stcore/health" || exit 1

# fileWatcherType=none: the deployed image is static (no hot-reload use case), and
# Streamlit's watcher otherwise walks every imported module's __path__ — including
# transformers' huge lazy-loaded optional model registry — logging a traceback per
# submodule it can't introspect (harmless, but alarming log noise).
CMD ["sh", "-c", "streamlit run src/docqa/app/main.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none"]
