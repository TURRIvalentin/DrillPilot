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
COPY docs/data_checksums.json ./docs/data_checksums.json

# USROP CSVs for the "cargar ejemplo real" button: downloaded and
# checksum-verified at BUILD time (same script as local dev setup), never
# committed to the repo (CC BY-NC-SA, no redistribution -- ADR-001) and
# never copied from the host (unlike docker-compose's local ./data mount,
# which ECS has no equivalent of) -- see
# docs/adr/008-frontend-sample-data-build-time-download.md.
# Placed before COPY frontend so this cache layer only invalidates when the
# download script or the pinned checksums change, not on every frontend/ edit.
RUN python ml/data/download_usrop.py

COPY frontend ./frontend
RUN uv pip install --system ".[ml,frontend]"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "frontend/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
