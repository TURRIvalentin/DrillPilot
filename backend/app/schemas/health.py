"""Response schema for GET /health."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' si el modelo esta cargado, 'degraded' si no")
    model_loaded: bool = Field(..., description="Si el pipeline de produccion ya fue cargado")
    model_uri: str = Field(..., description="URI del modelo en el registry de MLflow")
