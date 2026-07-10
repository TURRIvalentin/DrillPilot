"""LightGBM model for ROP (M4), tuned with Optuna via leave-one-well-out CV.

Per ADR-002, LightGBM is the single ML algorithm for the MVP (XGBoost/CatBoost are
out of scope). Per ADR-003, hyperparameter tuning uses leave-one-well-out CV over the
CV-pool (wells 1, 2, 4, 6) -- never the held-out test wells (0, 3, 5). Optuna optimizes
mean LOWO-CV MAE directly (not RMSE, per ADR-001's rationale on ROP near zero).

Consumes the M3 feature matrix (ml.features.pipeline.USROPFeatureTransformer output),
not the raw cleaned columns -- unlike ml.models.byoung_reduced, LightGBM uses the
engineered window features too.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml.training.cv import leave_one_well_out_splits

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAME = "LightGBM"

_EARLY_STOPPING_ROUNDS = 30
_MAX_ESTIMATORS_DURING_TUNING = 1000


def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, well_id: pd.Series) -> float:
    params: dict[str, Any] = {
        "objective": "mae",
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": _MAX_ESTIMATORS_DURING_TUNING,
        "random_state": 42,
        "verbosity": -1,
    }

    fold_maes: list[float] = []
    best_iterations: list[int] = []
    for train_pos, val_pos, _held_out_well in leave_one_well_out_splits(well_id):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[train_pos],
            y.iloc[train_pos],
            eval_set=[(X.iloc[val_pos], y.iloc[val_pos])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=_EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        preds = model.predict(X.iloc[val_pos])
        fold_maes.append(float(mean_absolute_error(y.iloc[val_pos], preds)))
        best_iterations.append(int(model.best_iteration_))

    trial.set_user_attr("best_iterations", best_iterations)
    return float(np.mean(fold_maes))


def tune_and_fit_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    well_id: pd.Series,
    n_trials: int = 20,
    random_state: int = 42,
) -> tuple[lgb.LGBMRegressor, optuna.Study, dict[str, Any]]:
    """Tune LightGBM via Optuna (objective: mean LOWO-CV MAE), then refit on the full
    CV-pool with the best hyperparameters. Returns (fitted_model, study, final_params).

    `n_estimators` for the final fit is the average `best_iteration_` (from early
    stopping) across the winning trial's 4 folds -- the final fit has no held-out fold
    of its own to early-stop against, since it trains on the entire CV-pool.
    """
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: _objective(trial, X, y, well_id), n_trials=n_trials)

    best_iterations = study.best_trial.user_attrs["best_iterations"]
    final_n_estimators = max(int(round(float(np.mean(best_iterations)))), 10)

    final_params: dict[str, Any] = {
        "objective": "mae",
        **study.best_params,
        "n_estimators": final_n_estimators,
        "random_state": random_state,
        "verbosity": -1,
    }

    logger.info("Mejores hiperparametros (LOWO-CV MAE=%.4f): %s", study.best_value, final_params)

    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X, y)
    return final_model, study, final_params
