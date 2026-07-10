"""Tests for the dummy (global mean) baseline."""

from __future__ import annotations

import pandas as pd

from ml.models.dummy_baseline import GlobalMeanBaseline


def test_predicts_train_mean_for_every_row() -> None:
    X = pd.DataFrame({"anything": [1, 2, 3]})
    y = pd.Series([10.0, 20.0, 30.0])

    model = GlobalMeanBaseline().fit(X, y)
    preds = model.predict(pd.DataFrame({"anything": [1, 2, 3, 4, 5]}))

    assert (preds == 20.0).all()
    assert len(preds) == 5


def test_ignores_X_content() -> None:
    y = pd.Series([5.0, 15.0])

    model1 = GlobalMeanBaseline().fit(pd.DataFrame({"a": [1, 2]}), y)
    model2 = GlobalMeanBaseline().fit(pd.DataFrame({"a": [999, -999]}), y)

    assert model1.mean_ == model2.mean_ == 10.0
