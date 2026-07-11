"""M7: FastAPI app. Loads the production pipeline once at startup (lifespan) and
shares it across every request -- the service itself is stateless per-request (see
docs/adr/004-inference-input-contract.md); only the already-fitted model is shared.

Run: uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.explain import router as explain_router
from backend.app.api.health import router as health_router
from backend.app.api.predict import router as predict_router
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    service = InferenceService()
    try:
        service.load()
        logger.info("Modelo de produccion cargado correctamente.")
    except Exception:
        # /health reports model_loaded=False and /predict, /explain return 503
        # (ModelNotLoadedError) until the registry is reachable -- the process itself
        # should not crash just because MLflow was unreachable at startup.
        logger.exception("No se pudo cargar el modelo de produccion al arrancar.")
    app.state.inference_service = service
    yield


app = FastAPI(title="DrillPilot API", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(health_router)
