"""Tests for the LightGBM + Optuna tuning wrapper. Uses tiny synthetic data and
n_trials=2 to keep this fast -- the real tuning run (n_trials=20 on the full CV-pool)
happens in ml/training/train_baselines.py, not in the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models.lightgbm_model import tune_and_fit_lightgbm


def _synthetic_data(
    n_per_well: int = 200, n_wells: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    frames = []
    for well in range(n_wells):
        frame = pd.DataFrame(
            {
                "f1": rng.uniform(0, 10, n_per_well),
                "f2": rng.uniform(0, 10, n_per_well),
                "well_id": well,
            }
        )
        frames.append(frame)
    X = pd.concat(frames, ignore_index=True)
    noise = np.random.default_rng(seed + 1).normal(0, 0.5, len(X))
    y = 5 + 2 * X["f1"] - 0.5 * X["f2"] + noise
    well_id = X.pop("well_id")
    return X, pd.Series(y), well_id


def test_tune_and_fit_returns_fitted_model_with_finite_predictions() -> None:
    X, y, well_id = _synthetic_data()

    model, _study, _params = tune_and_fit_lightgbm(X, y, well_id, n_trials=2)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert not np.isnan(preds).any()


def test_final_params_use_averaged_best_iteration_from_winning_trial() -> None:
    X, y, well_id = _synthetic_data()

    _model, study, params = tune_and_fit_lightgbm(X, y, well_id, n_trials=2)

    assert params["n_estimators"] >= 10
    assert params["objective"] == "mae"
    assert "best_iterations" in study.best_trial.user_attrs
    assert len(study.best_trial.user_attrs["best_iterations"]) == 3  # one per well/fold
