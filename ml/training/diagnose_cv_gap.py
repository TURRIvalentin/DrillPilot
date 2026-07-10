"""M4 diagnostics (post-hoc, does NOT retrain or replace the committed M4 models):
breaks the docs/m4_results.md CV-vs-test gap finding into two pieces requested before
deciding how to proceed to M5:

1. Per-fold LOWO-CV MAE for the already-tuned LightGBM (not just the mean Optuna
   optimized) -- using the exact hyperparameters already logged in MLflow, not a
   retune.
2. An ablation: retune LightGBM (same Optuna + LOWO-CV protocol, same n_trials) using
   only the 12 direct M3 features, excluding the 4 window features, and compare its
   test-set performance to the full-feature model already in docs/m4_metrics.json.

Run: python -m ml.training.diagnose_cv_gap
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from ml.evaluation.metrics import mae_report
from ml.features.dataset import load_combined_dataset
from ml.features.pipeline import DIRECT_FEATURE_COLUMNS, USROPFeatureTransformer
from ml.features.split import split_test_cv_pool
from ml.models.lightgbm_model import tune_and_fit_lightgbm
from ml.training.cv import leave_one_well_out_splits
from ml.training.train_baselines import EXPERIMENT_NAME

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = _REPO_ROOT / "docs" / "m4_diagnostics.json"

_TUNABLE_PARAM_KEYS = (
    "num_leaves",
    "learning_rate",
    "max_depth",
    "min_child_samples",
    "colsample_bytree",
    "subsample",
    "subsample_freq",
    "reg_alpha",
    "reg_lambda",
)


def _fetch_best_lightgbm_params() -> dict[str, Any]:
    """Pull the exact tuned hyperparameters from the already-logged MLflow run
    (source of truth) -- not retuned, not copy-pasted from a log transcript."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento MLflow '{EXPERIMENT_NAME}'.")
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.model_family = 'lightgbm'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "No se encontro una corrida de LightGBM en MLflow -- correr "
            "ml/training/train_baselines.py primero."
        )
    raw = runs[0].data.params
    return {
        "num_leaves": int(raw["num_leaves"]),
        "learning_rate": float(raw["learning_rate"]),
        "max_depth": int(raw["max_depth"]),
        "min_child_samples": int(raw["min_child_samples"]),
        "colsample_bytree": float(raw["colsample_bytree"]),
        "subsample": float(raw["subsample"]),
        "subsample_freq": int(raw["subsample_freq"]),
        "reg_alpha": float(raw["reg_alpha"]),
        "reg_lambda": float(raw["reg_lambda"]),
    }


def per_fold_cv_mae(
    X: pd.DataFrame, y: pd.Series, well_id: pd.Series, tunable_params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Re-run the exact LOWO-CV protocol used during Optuna tuning (n_estimators cap
    1000 + 30-round early stopping) for one fixed, already-chosen hyperparameter set,
    returning MAE per held-out well instead of just the mean Optuna optimized.
    """
    params = {
        "objective": "mae",
        **tunable_params,
        "n_estimators": 1000,
        "random_state": 42,
        "verbosity": -1,
    }
    results = []
    for train_pos, val_pos, held_out_well in leave_one_well_out_splits(well_id):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[train_pos],
            y.iloc[train_pos],
            eval_set=[(X.iloc[val_pos], y.iloc[val_pos])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds = model.predict(X.iloc[val_pos])
        mae = float(mean_absolute_error(y.iloc[val_pos], preds))
        results.append(
            {
                "held_out_well": held_out_well,
                "val_rows": int(len(val_pos)),
                "mae": mae,
                "best_iteration": int(model.best_iteration_),
            }
        )
        logger.info(
            "fold pozo=%d val_rows=%d MAE=%.4f best_iter=%d",
            held_out_well,
            len(val_pos),
            mae,
            model.best_iteration_,
        )
    return results


def run_ablation_without_window_features(n_trials: int = 15) -> dict[str, Any]:
    """Retune + evaluate LightGBM using only the 12 direct M3 features (no rolling/diff
    window features), identical Optuna + LOWO-CV protocol as the full-feature model, for
    a fair test-set comparison against docs/m4_metrics.json's "lightgbm" entry.
    """
    df = load_combined_dataset()
    cv_pool_df, test_df = split_test_cv_pool(df)

    transformer = USROPFeatureTransformer()
    X_cv_full = transformer.fit_transform(cv_pool_df)
    X_test_full = transformer.transform(test_df)

    direct_cols = list(DIRECT_FEATURE_COLUMNS)
    X_cv = X_cv_full[direct_cols]
    X_test = X_test_full[direct_cols]

    model, study, params = tune_and_fit_lightgbm(
        X_cv, cv_pool_df["ROP"], cv_pool_df["well_id"], n_trials=n_trials
    )
    test_preds = model.predict(X_test)
    test_report = mae_report(test_df["ROP"], test_preds, test_df["well_id"])

    return {
        "features_used": direct_cols,
        "cv_mae_lowo": study.best_value,
        "test_pooled": test_report.pooled,
        "test_by_well": test_report.by_well,
        "test_by_regime": test_report.by_regime,
        "best_params": params,
    }


def main(n_trials: int = 15) -> dict[str, Any]:
    df = load_combined_dataset()
    cv_pool_df, _test_df = split_test_cv_pool(df)
    transformer = USROPFeatureTransformer()
    X_cv = transformer.fit_transform(cv_pool_df)

    logger.info("=== Diagnostico 1: MAE por fold LOWO-CV (LightGBM ya tuneado) ===")
    tunable_params = _fetch_best_lightgbm_params()
    fold_results = per_fold_cv_mae(X_cv, cv_pool_df["ROP"], cv_pool_df["well_id"], tunable_params)
    mean_of_folds = sum(f["mae"] for f in fold_results) / len(fold_results)
    logger.info("Media de los 4 folds: %.4f (comparar contra cv_mae_lowo logueado)", mean_of_folds)

    logger.info("=== Diagnostico 2: ablation sin features de ventana ===")
    ablation_results = run_ablation_without_window_features(n_trials=n_trials)

    results = {
        "per_fold_cv_mae": fold_results,
        "per_fold_cv_mae_mean_check": mean_of_folds,
        "ablation_no_window_features": ablation_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultados guardados en %s", RESULTS_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
