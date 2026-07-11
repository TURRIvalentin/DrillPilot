"""Pydantic request schema for a single drilling reading -- one row of the history
window ADR-004 requires. Field names match ml.features.pipeline.DIRECT_FEATURE_COLUMNS
plus well_id exactly, so a list of Reading maps 1:1 onto the DataFrame columns the
loaded pipeline expects -- no separate renaming/translation layer to keep in sync.
Units per docs/data_dictionary.md.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Reading(BaseModel):
    well_id: int = Field(
        ...,
        description=(
            "Clave de agrupamiento de esta sesion/pozo. No necesita coincidir con "
            "ninguno de los 7 pozos de entrenamiento (0-6) -- ver ml/inference/predict.py."
        ),
    )
    MD: float = Field(..., description="Profundidad medida (m)")
    WOB: float = Field(..., description="Peso sobre el trepano (kkgf)")
    SPP: float = Field(..., description="Presion promedio en standpipe (kPa)")
    T: float = Field(..., description="Torque promedio en superficie (kN.m)")
    RPM: float = Field(..., description="Velocidad rotatoria promedio (rpm)")
    FR: float = Field(..., description="Caudal de lodo de entrada (L/min)")
    DS: float = Field(..., description="Densidad del lodo de entrada (g/cm3)")
    HD: float = Field(..., description="Diametro del hoyo/broca (mm)")
    HL: float = Field(..., description="Hookload promedio (kkgf)")
    VD: float = Field(..., description="Profundidad vertical verdadera (m)")
    GR: float = Field(..., description="Rayos gamma (gAPI)")
    gr_imputed: bool = Field(
        ...,
        description=(
            "Si el valor de GR de esta fila fue interpolado (regla de limpieza 3, M2) "
            "en vez de medido -- ver docs/feature_dictionary.md."
        ),
    )
