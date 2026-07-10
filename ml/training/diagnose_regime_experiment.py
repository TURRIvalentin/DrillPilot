"""M4 regime experiment (post-hoc diagnostic, does NOT retrain or replace the
committed M4 models): tests whether the CV-test gap can be closed by giving LightGBM
explicit regime information, as a targeted alternative to "just too few wells" --
requested before choosing between the 3 options in docs/m4_results.md.

1a. LightGBM (full CV-pool, same Optuna + 4-fold LOWO-CV protocol as M4) with one
    added feature: regime_score, P(regimen dominante) derived from MD and HD only
    (both real-time-available at inference -- you always know current depth and the
    bit/hole diameter in the string) via a small logistic regression fit on the
    CV-pool only. Never uses well_id.
2a. Two separate LightGBM models (own Optuna + LOWO-CV tuning each), trained only on
    the CV-pool wells of one regime -- dominante (2, 4) and atipico (1, 6) -- and
    evaluated only against the matching test wells (3, 5 and 0 respectively).

Run: python -m ml.training.diagnose_regime_experiment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.evaluation.metrics import DOMINANT_REGIME_WELL_IDS, mae_report
from ml.features.dataset import load_combined_dataset
from ml.features.pipeline import USROPFeatureTransformer
from ml.features.split import split_test_cv_pool
from ml.models.lightgbm_model import tune_and_fit_lightgbm

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = _REPO_ROOT / "docs" / "m4_regime_experiment.json"

# CV-pool wells by regime (subset of ADR-003's CV_POOL_WELL_IDS = {1, 2, 4, 6}).
CV_POOL_DOMINANT_WELLS: tuple[int, ...] = (2, 4)
CV_POOL_ATYPICAL_WELLS: tuple[int, ...] = (1, 6)
# Matching test wells by regime (subset of ADR-003's TEST_WELL_IDS = {0, 3, 5}).
TEST_DOMINANT_WELLS: tuple[int, ...] = (3, 5)
TEST_ATYPICAL_WELLS: tuple[int, ...] = (0,)


def fit_regime_score_classifier(cv_pool_df: pd.DataFrame) -> LogisticRegression:
    """Fit a small logistic regression predicting P(regimen dominante) from MD and HD
    alone -- both real-time-available drilling parameters (current depth and hole/bit
    diameter are always known), unlike well_id, which is bookkeeping with no meaning
    for a genuinely new well. Fit on the CV-pool only, class-balanced (CV-pool has far
    more dominant-regime rows than atypical: 99,353 vs 14,240).
    """
    is_dominant = cv_pool_df["well_id"].isin(DOMINANT_REGIME_WELL_IDS).astype(int)
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(cv_pool_df[["MD", "HD"]], is_dominant)
    return classifier


def add_regime_score_feature(
    X: pd.DataFrame, raw_df: pd.DataFrame, classifier: LogisticRegression
) -> pd.DataFrame:
    """Append `regime_score` = P(regimen dominante | MD, HD) to feature matrix `X`.
    `raw_df` must have the same row order/length as `X` (both derived from the same
    source frame without reordering)."""
    X = X.copy()
    X["regime_score"] = classifier.predict_proba(raw_df[["MD", "HD"]])[:, 1]
    return X


def run_regime_score_experiment(n_trials: int = 15) -> dict[str, Any]:
    """Experiment 1a: full CV-pool + regime_score feature, same protocol as M4."""
    df = load_combined_dataset()
    cv_pool_df, test_df = split_test_cv_pool(df)

    classifier = fit_regime_score_classifier(cv_pool_df)
    true_dominant = cv_pool_df["well_id"].isin(DOMINANT_REGIME_WELL_IDS).astype(int)
    predicted_dominant = classifier.predict(cv_pool_df[["MD", "HD"]])
    cv_pool_accuracy = float((predicted_dominant == true_dominant).mean())

    transformer = USROPFeatureTransformer()
    X_cv = add_regime_score_feature(transformer.fit_transform(cv_pool_df), cv_pool_df, classifier)
    X_test = add_regime_score_feature(transformer.transform(test_df), test_df, classifier)

    model, study, params = tune_and_fit_lightgbm(
        X_cv, cv_pool_df["ROP"], cv_pool_df["well_id"], n_trials=n_trials
    )
    test_preds = model.predict(X_test)
    test_report = mae_report(test_df["ROP"], test_preds, test_df["well_id"])

    return {
        "regime_classifier_cv_pool_accuracy": cv_pool_accuracy,
        "regime_classifier_coefficients": {
            "MD": float(classifier.coef_[0][0]),
            "HD": float(classifier.coef_[0][1]),
            "intercept": float(classifier.intercept_[0]),
        },
        "cv_mae_lowo": study.best_value,
        "test_pooled": test_report.pooled,
        "test_by_well": test_report.by_well,
        "test_by_regime": test_report.by_regime,
        "best_params": params,
    }


def run_regime_specific_models(
    n_trials: int = 15,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Experiment 1b: separate LightGBM per regime, trained/tuned only on that
    regime's CV-pool wells, evaluated only against that regime's test wells (i.e.
    assuming *oracle* / ground-truth regime routing -- see
    evaluate_realistic_routing for what happens with real classifier-based routing).

    Returns (json_serializable_results, {"dominante": fitted_model, "atipico": fitted_model}).
    """
    df = load_combined_dataset()
    cv_pool_df, test_df = split_test_cv_pool(df)
    transformer = USROPFeatureTransformer()

    results: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    for name, cv_wells, test_wells in (
        ("dominante", CV_POOL_DOMINANT_WELLS, TEST_DOMINANT_WELLS),
        ("atipico", CV_POOL_ATYPICAL_WELLS, TEST_ATYPICAL_WELLS),
    ):
        cv_subset = cv_pool_df[cv_pool_df["well_id"].isin(cv_wells)].reset_index(drop=True)
        test_subset = test_df[test_df["well_id"].isin(test_wells)].reset_index(drop=True)

        X_cv = transformer.fit_transform(cv_subset)
        X_test = transformer.transform(test_subset)

        model, study, params = tune_and_fit_lightgbm(
            X_cv, cv_subset["ROP"], cv_subset["well_id"], n_trials=n_trials
        )
        test_preds = model.predict(X_test)
        test_report = mae_report(test_subset["ROP"], test_preds, test_subset["well_id"])

        results[name] = {
            "cv_pool_wells": list(cv_wells),
            "test_wells": list(test_wells),
            "cv_pool_rows": int(len(cv_subset)),
            "n_lowo_folds": len(cv_wells),
            "cv_mae_lowo": study.best_value,
            "test_pooled": test_report.pooled,
            "test_by_well": test_report.by_well,
            "best_params": params,
        }
        fitted_models[name] = model
        logger.info(
            "regimen=%s: CV MAE=%.4f test pooled=%.4f", name, study.best_value, test_report.pooled
        )

    return results, fitted_models


def evaluate_realistic_routing(
    classifier: LogisticRegression, fitted_models: dict[str, Any]
) -> dict[str, Any]:
    """The honest version of experiment 1b: route each TEST row to the dominante or
    atipico specialized model using the classifier's *prediction* (MD, HD only) --
    not the ground-truth regime table from ADR-003, which a genuinely new well would
    not have. Reveals whether the oracle-routed result in `regime_specific_models` is
    actually achievable in a real deployment.
    """
    df = load_combined_dataset()
    _cv_pool_df, test_df = split_test_cv_pool(df)
    transformer = USROPFeatureTransformer()
    X_test = transformer.transform(test_df)

    predicted_dominant = classifier.predict(test_df[["MD", "HD"]])
    preds_dominante = fitted_models["dominante"].predict(X_test)
    preds_atipico = fitted_models["atipico"].predict(X_test)
    final_preds = np.where(predicted_dominant == 1, preds_dominante, preds_atipico)

    report = mae_report(test_df["ROP"], final_preds, test_df["well_id"])

    classifier_accuracy_by_well = {}
    true_dominant = test_df["well_id"].isin(DOMINANT_REGIME_WELL_IDS).astype(int)
    for well in sorted(test_df["well_id"].unique()):
        mask = test_df["well_id"] == well
        acc = float((predicted_dominant[mask] == true_dominant[mask]).mean())
        classifier_accuracy_by_well[str(int(well))] = acc

    return {
        "classifier_accuracy_by_well": classifier_accuracy_by_well,
        "test_pooled": report.pooled,
        "test_by_well": report.by_well,
        "test_by_regime": report.by_regime,
    }


def main(n_trials: int = 15) -> dict[str, Any]:
    logger.info("=== Experimento 1a: LightGBM + regime_score feature ===")
    regime_score_results = run_regime_score_experiment(n_trials=n_trials)

    logger.info("=== Experimento 1b: modelos separados por regimen (routing oraculo) ===")
    regime_specific_results, fitted_models = run_regime_specific_models(n_trials=n_trials)

    logger.info("=== Experimento 1b honesto: routing real via clasificador ===")
    df = load_combined_dataset()
    cv_pool_df, _test_df = split_test_cv_pool(df)
    classifier = fit_regime_score_classifier(cv_pool_df)
    realistic_routing_results = evaluate_realistic_routing(classifier, fitted_models)
    logger.info(
        "Routing real (clasificador): test pooled=%.4f", realistic_routing_results["test_pooled"]
    )

    results = {
        "regime_score_feature": regime_score_results,
        "regime_specific_models": regime_specific_results,
        "realistic_classifier_routed_system": realistic_routing_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultados guardados en %s", RESULTS_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
