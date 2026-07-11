"""POST /explain -- prediction + SHAP breakdown of the last reading in the window
(see docs/adr/005-shap-endpoint-design.md for why this is separate from /predict)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_inference_service
from backend.app.schemas.explain import ExplainRequest, ExplainResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    service: InferenceService = Depends(get_inference_service),
) -> ExplainResponse:
    return service.explain(request)
