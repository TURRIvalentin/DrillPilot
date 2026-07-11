# syntax=docker/dockerfile:1
#
# Frontend (Streamlit) image. Talks to the backend over HTTP only
# (DRILLPILOT_BACKEND_URL) -- never loads the model itself, so it does not need
# docker/model_artifact/ at all (see docs/adr/006-model-packaging-deploy.md,
# which is a backend-only concern).
#
# Build context is the repo root (see docker-compose.yml), not frontend/, because
# this image also needs the ml/ package (ml.features.pipeline/.dataset --
# feature-column metadata and the local sample-data loader, not the model).

FROM python:3.12-slim AS frontend

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY ml ./ml
COPY frontend ./frontend
RUN uv pip install --system ".[ml,frontend]"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "frontend/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
