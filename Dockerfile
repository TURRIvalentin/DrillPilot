# syntax=docker/dockerfile:1
#
# Backend (FastAPI) image. Multi-stage: the builder stage has the compiler
# toolchain lightgbm/shap may need to build from source; the runtime stage only
# gets the resulting venv, never the build tools themselves.
#
# The model artifact copied below (docker/model_artifact/) is NOT built here --
# it is exported once, locally, via `python -m ml.inference.export_model` (which
# needs the local MLflow tracking store) and committed to the repo. The image
# never talks to a tracking server, at build time or at runtime -- see
# docs/adr/006-model-packaging-deploy.md for why, and for the exact pinned
# run_id (ml/inference/export_model.py::PINNED_PRODUCTION_RUN_ID) this artifact
# corresponds to.

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml ./
COPY ml ./ml
COPY backend ./backend
RUN uv pip install --python /opt/venv/bin/python ".[ml,api]"


FROM python:3.12-slim AS backend

WORKDIR /app

# libgomp1: LightGBM's OpenMP runtime dependency -- needed to run a trained model,
# even though the compiler itself (build-essential, builder stage only) is not.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY ml ./ml
COPY backend ./backend
COPY docker/model_artifact ./docker/model_artifact

ENV DRILLPILOT_MODEL_URI=/app/docker/model_artifact
ENV DRILLPILOT_LOG_LEVEL=INFO

EXPOSE 8000

# Real healthcheck: the process can be alive and still serving with no model
# loaded (see backend/app/main.py's lifespan -- a failed load does not crash the
# process). This checks that /health actually reports model_loaded=true, not just
# that something answers on the port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import json,sys,urllib.request; \
  r=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=3)); \
  sys.exit(0 if r.get('model_loaded') else 1)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
