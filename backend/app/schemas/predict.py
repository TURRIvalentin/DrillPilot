"""Request/response schemas for POST /predict (ADR-004: window contract, ADR-005:
no SHAP on this endpoint -- see backend/app/schemas/explain.py for that).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.schemas.reading import Reading


class PredictionRequest(BaseModel):
    readings: list[Reading] = Field(
        ...,
        min_length=1,
        description=(
            "Ventana de lecturas de un mismo pozo/sesion, ordenadas por MD ascendente. "
            "Minimo 1 fila aceptada; recomendado >=10 para que las features de ventana "
            "(WOB_rolling_mean_10, RPM_rolling_mean_10) no queden en su regimen "
            "degradado de poco historial -- ver docs/adr/004-inference-input-contract.md."
        ),
    )


class PredictionItem(BaseModel):
    md: float = Field(..., description="Profundidad medida (m) de esta lectura")
    predicted_rop: float = Field(..., description="ROP predicho (m/h)")
    known_limitation_zone: bool = Field(
        ...,
        description=(
            "True si md cae dentro de ml.evaluation.metrics.KNOWN_LIMITATION_MD_RANGE_M "
            "(634-988 m), la banda de profundidad confirmada como zona de alto error "
            "(sin cobertura de ningun regimen en el CV-pool, ver docs/m6_results.md). "
            "Senal de baja confianza, no un descarte de la prediccion."
        ),
    )


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem] = Field(
        ..., description="Una prediccion por lectura recibida, en el mismo orden."
    )
