"""SHAP explainability (M5) for the M4 candidate LightGBM model.

Global + local explanations per the original roadmap, plus a targeted dependence
check on MD and HD -- given everything M4's regime diagnostics turned up about a
depth-related regime boundary (docs/m4_results.md), this checks whether SHAP shows
a behavior break in the model consistent with that boundary. No particular result is
assumed going in; whatever comes out is reported as-is in docs/m5_results.md.

Model: the M4 candidate tagged m5_candidate=true in MLflow (run CANDIDATE_RUN_ID,
"lightgbm-tuned" in experiment drillpilot-m4-baselines) -- loaded from the actual
logged artifact, never retrained.

Run: python -m ml.explainability.shap_explain
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from ml.evaluation.metrics import DOMINANT_REGIME_WELL_IDS  # noqa: E402
from ml.features.dataset import load_combined_dataset  # noqa: E402
from ml.features.pipeline import USROPFeatureTransformer  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _REPO_ROOT / "docs" / "shap"
RESULTS_PATH = _REPO_ROOT / "docs" / "m5_shap_summary.json"

# Tagged m5_candidate=true in MLflow -- the original tuned LightGBM from
# ml/training/train_baselines.py, not any of the M4 diagnostic variants.
CANDIDATE_RUN_ID = "cbab6ab7cbbe435da1fc63a9c23a5542"

# MD band with zero CV-pool coverage from either regime (docs/m4_results.md, regime
# experiment): atypical CV-pool wells (1, 6) top out at 634 m, dominant CV-pool wells
# (2, 4) start at 988 m. Test well 0 (atypical) reaches into this band (up to 1206 m)
# -- exactly where the M4 regime router failed. Marked on the MD dependence plot.
REGIME_GAP_MD_RANGE_M: tuple[float, float] = (634.0, 988.0)


def load_candidate_model() -> Any:
    """Load the M4 candidate LightGBM from its logged MLflow artifact (not retrained)."""
    return mlflow.lightgbm.load_model(f"runs:/{CANDIDATE_RUN_ID}/model")


def build_explanation_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full USROP dataset (all 7 wells), transformed to the model's feature matrix.

    Returns (X_full, raw_full): X_full is what the model sees; raw_full carries
    MD/HD/well_id (same row order) for dependence-plot regime annotation -- never fed
    to the model itself.
    """
    df = load_combined_dataset()
    transformer = USROPFeatureTransformer()
    X_full = transformer.transform(df)
    return X_full, df


def regime_label(well_id: pd.Series) -> pd.Series:
    """'dominante' / 'atipico' label per well_id, for plot annotation only -- never a
    model feature."""
    return well_id.isin(DOMINANT_REGIME_WELL_IDS).map({True: "dominante", False: "atipico"})


def compute_shap_values(model: Any, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def verify_shap_additivity(
    explanation: shap.Explanation, model: Any, X: pd.DataFrame, n_samples: int = 500
) -> float:
    """Sanity check: sum(shap_values) + expected_value should equal the model's raw
    prediction for each row (SHAP's additivity property). Returns the max absolute
    discrepancy across a random sample of rows -- should be ~0.
    """
    sample_idx = np.random.default_rng(42).choice(
        len(X), size=min(n_samples, len(X)), replace=False
    )
    preds = model.predict(X.iloc[sample_idx])
    reconstructed = explanation.values[sample_idx].sum(axis=1) + explanation.base_values[sample_idx]
    return float(np.max(np.abs(preds - reconstructed)))


def plot_global_summary(explanation: shap.Explanation, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "global_importance_bar.png", dpi=120)
    plt.close()

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "global_beeswarm.png", dpi=120)
    plt.close()


def plot_local_examples(
    explanation: shap.Explanation, row_positions: dict[str, int], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, position in row_positions.items():
        plt.figure()
        shap.plots.waterfall(explanation[position], show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"local_{label}.png", dpi=120)
        plt.close()


def plot_regime_dependence(
    explanation: shap.Explanation,
    X: pd.DataFrame,
    raw: pd.DataFrame,
    feature: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_idx = list(X.columns).index(feature)
    shap_vals = explanation.values[:, feature_idx]
    feature_vals = X[feature].to_numpy()
    regimes = regime_label(raw["well_id"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for regime, color in (("dominante", "#2b6cb0"), ("atipico", "#c05621")):
        mask = (regimes == regime).to_numpy()
        ax.scatter(feature_vals[mask], shap_vals[mask], s=6, alpha=0.4, label=regime, color=color)
    if feature == "MD":
        ax.axvspan(
            *REGIME_GAP_MD_RANGE_M,
            color="grey",
            alpha=0.15,
            label="gap sin cobertura en CV-pool",
        )
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value ({feature})")
    ax.set_title(f"Dependence plot: {feature}, coloreado por regimen")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"dependence_{feature}.png", dpi=120)
    plt.close(fig)


def main() -> dict[str, Any]:
    model = load_candidate_model()
    X_full, raw_full = build_explanation_dataset()

    logger.info("Calculando SHAP values sobre %d filas (7 pozos)...", len(X_full))
    explanation = compute_shap_values(model, X_full)

    additivity_error = verify_shap_additivity(explanation, model, X_full)
    logger.info("Chequeo de aditividad SHAP: error maximo=%.6f (deberia ser ~0)", additivity_error)

    plot_global_summary(explanation, OUTPUT_DIR)

    test_wells = (0, 3, 5)
    row_positions = {}
    for well in test_wells:
        position = raw_full.index.get_loc(raw_full.index[raw_full["well_id"] == well][0])
        row_positions[f"well_{well}"] = int(position)
    plot_local_examples(explanation, row_positions, OUTPUT_DIR / "local")

    plot_regime_dependence(explanation, X_full, raw_full, "MD", OUTPUT_DIR)
    plot_regime_dependence(explanation, X_full, raw_full, "HD", OUTPUT_DIR)

    mean_abs_shap = {
        col: float(np.mean(np.abs(explanation.values[:, i])))
        for i, col in enumerate(X_full.columns)
    }
    ranked = dict(sorted(mean_abs_shap.items(), key=lambda kv: -kv[1]))

    results = {
        "candidate_run_id": CANDIDATE_RUN_ID,
        "n_rows_explained": len(X_full),
        "additivity_max_error": additivity_error,
        "mean_abs_shap_by_feature": ranked,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultados guardados en %s", RESULTS_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
