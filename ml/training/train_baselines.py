"""Train and MLflow-track the three M4 baselines: dummy, Bourgoyne & Young reducido
(4/8 terminos), and LightGBM. Evaluates each on the ADR-003 test set (wells 0, 3, 5)
with pooled + per-well + per-regime MAE, per ADR-003's reporting requirement, and
writes the combined results to docs/m4_metrics.json (source of truth for
docs/m4_results.md -- not hand-typed).

Run: python -m ml.training.train_baselines
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd

from ml.evaluation.metrics import MaeReport, mae_report
from ml.features.dataset import load_combined_dataset
from ml.features.pipeline import USROPFeatureTransformer
from ml.features.split import split_test_cv_pool
from ml.models.byoung_reduced import MODEL_NAME as BYOUNG_MODEL_NAME
from ml.models.byoung_reduced import BourgoyneYoungReduced
from ml.models.dummy_baseline import GlobalMeanBaseline
from ml.models.lightgbm_model import MODEL_NAME as LIGHTGBM_MODEL_NAME
from ml.models.lightgbm_model import tune_and_fit_lightgbm
from ml.training.cv import leave_one_well_out_splits

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "drillpilot-m4-baselines"
_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = _REPO_ROOT / "docs" / "m4_metrics.json"


def _report_to_dict(report: MaeReport, cv_mae: float) -> dict[str, Any]:
    return {
        "cv_mae_lowo": cv_mae,
        "test_pooled": report.pooled,
        "test_by_well": report.by_well,
        "test_by_regime": report.by_regime,
    }


def _dummy_lowo_cv_mae(cv_pool_df: pd.DataFrame) -> float:
    """Fair LOWO-CV MAE for the dummy, under the exact same protocol as B&Y/LightGBM:
    fit the mean on 3 CV-pool wells, predict the 4th, average across the 4 folds. Used
    as context in docs/m4_results.md to show the dummy's *final test* competitiveness
    is not because it "learned" as much as LightGBM in-distribution (LightGBM beats it
    by a wide margin here) -- see the M4 report for the full discussion.
    """
    y = cv_pool_df["ROP"]
    fold_maes = []
    for train_pos, val_pos, _held_out in leave_one_well_out_splits(cv_pool_df["well_id"]):
        train_mean = y.iloc[train_pos].mean()
        preds = pd.Series(train_mean, index=y.iloc[val_pos].index)
        fold_maes.append(float((y.iloc[val_pos] - preds).abs().mean()))
    return float(pd.Series(fold_maes).mean())


def train_dummy(
    cv_pool_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[GlobalMeanBaseline, MaeReport, float]:
    with mlflow.start_run(run_name="dummy-global-mean"):
        mlflow.set_tag("model_family", "dummy")
        mlflow.set_tag("model_display_name", "Dummy (media global)")
        model = GlobalMeanBaseline().fit(cv_pool_df, cv_pool_df["ROP"])

        cv_mae = _dummy_lowo_cv_mae(cv_pool_df)
        mlflow.log_param("strategy", "global_mean_blind_to_regime")
        mlflow.log_param("train_mean_rop", model.mean_)
        mlflow.log_metric("cv_mae_lowo", cv_mae)

        test_preds = model.predict(test_df)
        test_report = mae_report(test_df["ROP"], test_preds, test_df["well_id"])
        mlflow.log_metrics(test_report.to_flat_metrics("test"))
        # Custom estimator (not a stock sklearn class) -- skops' default trusted-types
        # allowlist rejects it; cloudpickle is safe here since we fully control this code.
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")

    return model, test_report, cv_mae


def train_byoung(
    cv_pool_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[BourgoyneYoungReduced, MaeReport, float]:
    with mlflow.start_run(run_name="byoung-reduced-4-8-terminos"):
        mlflow.set_tag("model_family", "byoung_reduced")
        mlflow.set_tag("model_display_name", BYOUNG_MODEL_NAME)
        model = BourgoyneYoungReduced().fit(cv_pool_df, cv_pool_df["ROP"])

        mlflow.log_param("wob_threshold", model.wob_threshold_)
        mlflow.log_param("wob_threshold_grid", list(model.wob_threshold_grid))
        mlflow.log_param("a1", model.a1_)
        mlflow.log_param("a2", model.a2_)
        mlflow.log_param("a5", model.a5_)
        mlflow.log_param("a6", model.a6_)
        mlflow.log_metric("cv_mae_lowo", model.cv_mae_)
        mlflow.log_metric("x5_clipped_fraction", model.clipped_fraction_)
        mlflow.log_metric("x6_rpm_clipped_fraction", model.rpm_clipped_fraction_)

        test_preds = model.predict(test_df)
        test_report = mae_report(test_df["ROP"], test_preds, test_df["well_id"])
        mlflow.log_metrics(test_report.to_flat_metrics("test"))
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")

    return model, test_report, model.cv_mae_


def train_lightgbm(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    well_id_cv: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    well_id_test: pd.Series,
    n_trials: int,
) -> tuple[Any, MaeReport, float]:
    with mlflow.start_run(run_name="lightgbm-tuned"):
        mlflow.set_tag("model_family", "lightgbm")
        mlflow.set_tag("model_display_name", LIGHTGBM_MODEL_NAME)
        model, study, params = tune_and_fit_lightgbm(X_cv, y_cv, well_id_cv, n_trials=n_trials)

        mlflow.log_params(params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metric("cv_mae_lowo", study.best_value)

        test_preds = model.predict(X_test)
        test_report = mae_report(y_test, test_preds, well_id_test)
        mlflow.log_metrics(test_report.to_flat_metrics("test"))
        mlflow.lightgbm.log_model(model, "model")

    return model, test_report, study.best_value


def main(n_trials: int = 20) -> dict[str, Any]:
    """Train and MLflow-log all 3 M4 baselines, write docs/m4_metrics.json, return it."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_combined_dataset()
    cv_pool_df, test_df = split_test_cv_pool(df)
    logger.info(
        "CV-pool: %d filas (pozos %s)", len(cv_pool_df), sorted(cv_pool_df["well_id"].unique())
    )
    logger.info("Test: %d filas (pozos %s)", len(test_df), sorted(test_df["well_id"].unique()))

    logger.info("Entrenando dummy...")
    _dummy_model, dummy_report, dummy_cv_mae = train_dummy(cv_pool_df, test_df)

    logger.info("Entrenando Bourgoyne & Young reducido...")
    _byoung_model, byoung_report, byoung_cv_mae = train_byoung(cv_pool_df, test_df)

    logger.info("Generando features (M3) para LightGBM...")
    transformer = USROPFeatureTransformer()
    X_cv = transformer.fit_transform(cv_pool_df)
    X_test = transformer.transform(test_df)

    logger.info("Entrenando LightGBM (Optuna, n_trials=%d)...", n_trials)
    _lgbm_model, lgbm_report, lgbm_cv_mae = train_lightgbm(
        X_cv,
        cv_pool_df["ROP"],
        cv_pool_df["well_id"],
        X_test,
        test_df["ROP"],
        test_df["well_id"],
        n_trials=n_trials,
    )

    results = {
        "dummy": _report_to_dict(dummy_report, dummy_cv_mae),
        "byoung_reduced": _report_to_dict(byoung_report, byoung_cv_mae),
        "lightgbm": _report_to_dict(lgbm_report, lgbm_cv_mae),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultados guardados en %s", RESULTS_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
