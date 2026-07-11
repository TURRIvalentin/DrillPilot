"""M7: thin orchestration between the FastAPI layer and ml.inference/ml.explainability.
Holds the production pipeline (loaded once, see backend/app/main.py's lifespan) and
reuses it for both /predict and /explain -- no feature or model logic duplicated here,
per the project's original architecture, ADR-004 (input contract) and ADR-005
(latency-driven endpoint split).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import shap

from backend.app.core.exceptions import ModelNotLoadedError
from backend.app.schemas.explain import ExplainRequest, ExplainResponse, FeatureContribution
from backend.app.schemas.predict import PredictionItem, PredictionRequest, PredictionResponse
from backend.app.schemas.reading import Reading
from ml.evaluation.metrics import is_in_known_limitation_zone
from ml.features.pipeline import is_insufficient_history
from ml.inference.predict import load_production_model, predict_rop


def readings_to_frame(readings: list[Reading]) -> pd.DataFrame:
    """Convert the request's list of Reading into the DataFrame shape
    ml.features.pipeline.USROPFeatureTransformer expects -- field names already match
    1:1 (see backend/app/schemas/reading.py), so this is just a list-of-models to
    DataFrame conversion, not a translation layer."""
    return pd.DataFrame([r.model_dump() for r in readings])


class InferenceService:
    """`model` is `None` until `load()` runs (called from the app's lifespan at
    startup); the SHAP explainer is built lazily on first `explain()` call so
    `/predict` never pays for it (ADR-005)."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._explainer: shap.TreeExplainer | None = None

    def load(self) -> None:
        if self._model is None:
            self._model = load_production_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if self._model is None:
            raise ModelNotLoadedError("El modelo de produccion todavia no fue cargado.")

        history = readings_to_frame(request.readings)
        preds = predict_rop(history, model=self._model)

        items = [
            PredictionItem(
                md=float(md),
                predicted_rop=float(pred),
                known_limitation_zone=is_in_known_limitation_zone(float(md)),
            )
            for md, pred in zip(history["MD"], preds, strict=True)
        ]
        return PredictionResponse(
            predictions=items,
            insufficient_history=is_insufficient_history(len(request.readings)),
        )

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Explain the LAST reading of the window only (ADR-005) -- SHAP cost scales
        with rows explained, and a caller wanting the current prediction's reasoning
        needs one point, not the whole window."""
        if self._model is None:
            raise ModelNotLoadedError("El modelo de produccion todavia no fue cargado.")

        history = readings_to_frame(request.readings)
        features_step = self._model.named_steps["features"]
        model_step = self._model.named_steps["model"]

        X = features_step.transform(history)
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(model_step)

        last_row = X.iloc[[-1]]
        explanation = self._explainer(last_row)
        predicted_rop = float(model_step.predict(last_row)[0])
        md = float(history["MD"].iloc[-1])

        contributions = [
            FeatureContribution(
                feature=col,
                value=float(last_row.iloc[0, i]),
                shap_value=float(explanation.values[0, i]),
            )
            for i, col in enumerate(X.columns)
        ]

        return ExplainResponse(
            md=md,
            predicted_rop=predicted_rop,
            known_limitation_zone=is_in_known_limitation_zone(md),
            insufficient_history=is_insufficient_history(len(request.readings)),
            base_value=float(explanation.base_values[0]),
            contributions=contributions,
        )
