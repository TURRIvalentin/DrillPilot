"""Tests for ml.inference.predict -- the single features+model inference boundary.
Uses a trivial fitted Pipeline (features + GlobalMeanBaseline, not LightGBM) so these
stay fast and offline -- no MLflow registry access needed when a model is passed in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.features.pipeline import USROPFeatureTransformer
from ml.inference.predict import predict_rop
from ml.models.dummy_baseline import GlobalMeanBaseline


def _history_frame(n: int = 5, well_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [well_id] * n,
            "MD": [float(i) for i in range(n)],
            "WOB": [5.0] * n,
            "SPP": [1000.0] * n,
            "T": [1.0] * n,
            "RPM": [100.0] * n,
            "FR": [2000.0] * n,
            "DS": [1.2] * n,
            "HD": [300.0] * n,
            "HL": [90.0] * n,
            "VD": [float(i) for i in range(n)],
            "GR": [50.0] * n,
            "gr_imputed": [False] * n,
        }
    )


def _fitted_pipeline() -> Pipeline:
    history = _history_frame()
    model = GlobalMeanBaseline().fit(history, pd.Series([20.0] * len(history)))
    return Pipeline([("features", USROPFeatureTransformer()), ("model", model)])


def test_predict_rop_returns_one_prediction_per_row() -> None:
    history = _history_frame(6)
    pipeline = _fitted_pipeline()

    preds = predict_rop(history, model=pipeline)

    assert len(preds) == len(history)
    assert np.all(preds == 20.0)


def test_predict_rop_does_not_touch_the_registry_when_model_is_passed(monkeypatch) -> None:
    def _boom() -> Any:
        raise AssertionError("no deberia intentar cargar desde el registry")

    monkeypatch.setattr("ml.inference.predict.load_production_model", _boom)
    history = _history_frame(3)
    pipeline = _fitted_pipeline()

    predict_rop(history, model=pipeline)  # no debe lanzar


def test_predict_rop_raises_on_missing_columns() -> None:
    history = _history_frame(3).drop(columns=["HD"])
    pipeline = _fitted_pipeline()

    with pytest.raises(ValueError):
        predict_rop(history, model=pipeline)


def test_predict_rop_well_id_is_purely_a_grouping_key_not_a_training_well() -> None:
    history = _history_frame(4, well_id=999)  # not one of the 7 training wells
    pipeline = _fitted_pipeline()

    preds = predict_rop(history, model=pipeline)

    assert len(preds) == 4
