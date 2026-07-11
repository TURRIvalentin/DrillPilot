"""POST /predict -- fast path, no SHAP (see docs/adr/005-shap-endpoint-design.md)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_inference_service
from backend.app.schemas.predict import PredictionRequest, PredictionResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    return service.predict(request)
