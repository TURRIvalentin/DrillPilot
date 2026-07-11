"""GET /health -- reports whether the production model finished loading."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_inference_service
from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_uri=settings.model_uri,
    )
