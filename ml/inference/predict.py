"""M6: production inference -- loads the single combined artifact (features +
model, see ml.training.promote_model) aliased "production" in the MLflow registry,
and predicts ROP from raw drilling readings. This module never reimplements feature
logic: the loaded artifact IS the features+model boundary fixed in the project's
original architecture, so there is nothing to duplicate here or in any future backend.

Input is a per-well/session HISTORY (a DataFrame sorted by MD ascending), not one
isolated row: the window features (WOB_rolling_mean, RPM_rolling_mean) need
preceding rows of the same well to be computed correctly -- see
docs/feature_dictionary.md. `predict_rop` returns one prediction per row passed in;
a caller wanting "the current ROP" uses the last one.

`well_id` in the input is purely a grouping key for the rolling-window features
computed inside this call -- it does not need to match one of the 7 training wells
(0-6). Any consistent identifier for "this session" works (e.g. a fixed constant, or
a UUID per drilling session) -- it is never used as a model feature (see
docs/feature_dictionary.md, "well_id" exclusion).
"""

from __future__ import annotations

from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd

MODEL_URI = "models:/drillpilot-rop@production"


def load_production_model(model_uri: str = MODEL_URI) -> Any:
    """Load the combined features+model artifact from `model_uri`.

    Defaults to the "production" alias in the MLflow registry (dev/local use, needs
    a reachable tracking store). In a deployed container there is no tracking store
    at runtime (see docs/adr/006-model-packaging-deploy.md) -- the caller passes a
    plain local path there instead (e.g. DRILLPILOT_MODEL_URI, see
    backend/app/core/config.py), which mlflow.sklearn.load_model() also accepts.
    """
    return mlflow.sklearn.load_model(model_uri)


def predict_rop(history: pd.DataFrame, model: Any | None = None) -> np.ndarray:
    """Predict ROP for every row of `history` (sorted ascending by MD, single
    well/session -- see module docstring).

    Column validation and feature engineering are delegated entirely to the loaded
    pipeline (ml.features.pipeline.USROPFeatureTransformer) -- this function does not
    duplicate that logic, so it cannot drift from it.

    Pass `model` explicitly (e.g. loaded once at FastAPI startup) to avoid a registry
    round trip on every call; omitted, it loads the production model fresh.
    """
    model = model if model is not None else load_production_model()
    return np.asarray(model.predict(history))
