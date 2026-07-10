"""M4 diagnostics (post-hoc, does NOT retrain or replace the committed M4 models):
breaks the docs/m4_results.md CV-vs-test gap finding into pieces requested before
deciding how to proceed to M5:

1. Per-fold LOWO-CV MAE for the already-tuned LightGBM (not just the mean Optuna
   optimized) -- using the exact hyperparameters already logged in MLflow, not a
   retune.
2. An ablation: retune LightGBM (same Optuna + LOWO-CV protocol, same n_trials) using
   only the 12 direct M3 features, excluding the 4 window features, and compare its
   test-set performance to the full-feature model already in docs/m4_metrics.json.
3. (Hipotesis 3) A fixed, deliberately conservative LightGBM (no Optuna at all), and
   whether Optuna's own trial ranking during M4 tuning was dominated by one LOWO-CV
   fold (well 1) -- both requested before choosing between the 3 options in
   docs/m4_results.md's original recommendation.

Run: python -m ml.training.diagnose_cv_gap
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
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

# Deliberately conservative, hand-chosen (NOT Optuna-tuned) hyperparameters for
# Hipotesis 3: shallow trees, few leaves, large min_child_samples (forces each leaf to
# summarize many rows instead of memorizing a handful from one well), and reduced
# feature/bagging fractions. learning_rate is not part of the user's conservative
# recipe -- filled in at a moderate, unremarkable default (0.05) so it does not
# confound the comparison.
CONSERVATIVE_FIXED_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 1000,
    "colsample_bytree": 0.75,
    "subsample": 0.75,
    "subsample_freq": 1,
    "learning_rate": 0.05,
}

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


def train_fixed_conservative_lightgbm(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    well_id_cv: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    well_id_test: pd.Series,
) -> dict[str, Any]:
    """Train LightGBM with CONSERVATIVE_FIXED_PARAMS (no Optuna) on the full 16-feature
    matrix, using the exact same LOWO-CV protocol as the tuned model for the CV MAE,
    then refit on the full CV-pool (n_estimators = mean best_iteration across folds)
    and evaluate on test with the same pooled/by-well/by-regime report as every other
    M4 model.
    """
    fold_results = per_fold_cv_mae(X_cv, y_cv, well_id_cv, CONSERVATIVE_FIXED_PARAMS)
    cv_mae = float(np.mean([f["mae"] for f in fold_results]))
    final_n_estimators = max(
        int(round(float(np.mean([f["best_iteration"] for f in fold_results])))), 10
    )

    final_params = {
        "objective": "mae",
        **CONSERVATIVE_FIXED_PARAMS,
        "n_estimators": final_n_estimators,
        "random_state": 42,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_cv, y_cv)
    test_preds = model.predict(X_test)
    test_report = mae_report(y_test, test_preds, well_id_test)

    return {
        "params": final_params,
        "per_fold_cv_mae": fold_results,
        "cv_mae_lowo": cv_mae,
        "test_pooled": test_report.pooled,
        "test_by_well": test_report.by_well,
        "test_by_regime": test_report.by_regime,
    }


def analyze_optuna_trial_fold_bias(
    X_cv: pd.DataFrame, y_cv: pd.Series, well_id_cv: pd.Series, n_trials: int = 15
) -> dict[str, Any]:
    """Reconstruct the exact M4 Optuna study (same seed, same search space, same
    n_trials -- deterministic, not a new/different search) and check whether one
    LOWO-CV fold dominated which trials Optuna preferred.

    For each fold (held-out well), reports the per-trial MAE range/std across all
    `n_trials` trials, and its Pearson correlation with the trial's overall objective
    value (mean across folds) -- the value Optuna actually ranks trials by. A fold
    whose per-trial MAE varies a lot AND correlates strongly with the overall ranking
    is a fold that disproportionately drove which hyperparameters "won".
    """
    _model, study, _params = tune_and_fit_lightgbm(X_cv, y_cv, well_id_cv, n_trials=n_trials)

    trials = [t for t in study.trials if t.value is not None]
    held_out_wells = trials[0].user_attrs["held_out_wells"]
    overall_values = np.array([t.value for t in trials])

    per_fold: dict[str, dict[str, float]] = {}
    for i, well in enumerate(held_out_wells):
        fold_values = np.array([t.user_attrs["fold_maes"][i] for t in trials])
        correlation = float(np.corrcoef(fold_values, overall_values)[0, 1])
        per_fold[str(well)] = {
            "mean": float(fold_values.mean()),
            "std": float(fold_values.std()),
            "min": float(fold_values.min()),
            "max": float(fold_values.max()),
            "range": float(fold_values.max() - fold_values.min()),
            "correlation_with_trial_ranking": correlation,
        }

    return {
        "n_trials": n_trials,
        "held_out_wells": held_out_wells,
        "per_fold": per_fold,
        "winning_trial_value": study.best_value,
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


def main_hypothesis_3(n_trials: int = 15) -> dict[str, Any]:
    """Hipotesis 3: fixed conservative LightGBM (no Optuna) + fold-bias analysis of
    Optuna's own trial ranking. Merges into the existing docs/m4_diagnostics.json
    (diagnostics 1 and 2 stay untouched)."""
    df = load_combined_dataset()
    cv_pool_df, test_df = split_test_cv_pool(df)
    transformer = USROPFeatureTransformer()
    X_cv = transformer.fit_transform(cv_pool_df)
    X_test = transformer.transform(test_df)

    logger.info("=== Hipotesis 3a: LightGBM con hiperparametros fijos conservadores ===")
    conservative_results = train_fixed_conservative_lightgbm(
        X_cv,
        cv_pool_df["ROP"],
        cv_pool_df["well_id"],
        X_test,
        test_df["ROP"],
        test_df["well_id"],
    )
    logger.info(
        "Conservador: CV MAE=%.4f, test pooled=%.4f",
        conservative_results["cv_mae_lowo"],
        conservative_results["test_pooled"],
    )

    logger.info("=== Hipotesis 3b: sesgo por fold en la seleccion de Optuna ===")
    fold_bias_results = analyze_optuna_trial_fold_bias(
        X_cv, cv_pool_df["ROP"], cv_pool_df["well_id"], n_trials=n_trials
    )

    existing: dict[str, Any] = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    existing["conservative_fixed_lightgbm"] = conservative_results
    existing["optuna_trial_fold_bias"] = fold_bias_results
    RESULTS_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultados guardados en %s", RESULTS_PATH)
    return existing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
