"""Request/response schemas for POST /explain (ADR-005: separate from /predict
because SHAP costs ~3.1x the plain prediction, p50=14.2ms vs p95=21.0ms measured
in M5). Explains only the LAST reading of the window -- see ADR-005 alternative 2
for why the whole window is not explained in one call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.schemas.predict import PredictionRequest

# Same request shape as /predict -- the client sends the same window either way.
ExplainRequest = PredictionRequest


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Nombre de la feature (post feature engineering)")
    value: float = Field(..., description="Valor de la feature para esta lectura")
    shap_value: float = Field(..., description="Contribucion SHAP de esta feature a la prediccion")


class ExplainResponse(BaseModel):
    md: float = Field(
        ..., description="Profundidad medida (m) de la lectura explicada (la ultima de la ventana)"
    )
    predicted_rop: float = Field(..., description="ROP predicho (m/h)")
    known_limitation_zone: bool = Field(
        ...,
        description=(
            "True si md cae dentro de ml.evaluation.metrics.KNOWN_LIMITATION_MD_RANGE_M "
            "(634-988 m) -- ver PredictionItem.known_limitation_zone."
        ),
    )
    base_value: float = Field(
        ...,
        description="Valor base de SHAP (prediccion promedio del modelo antes de las contribuciones)",
    )
    contributions: list[FeatureContribution] = Field(
        ...,
        description=(
            "Contribucion SHAP de cada feature a la prediccion de la ultima lectura. "
            "sum(shap_value) + base_value == predicted_rop (propiedad de aditividad de SHAP, "
            "ver ml/explainability/shap_explain.py::verify_shap_additivity)."
        ),
    )
