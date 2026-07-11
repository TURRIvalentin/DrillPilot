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
from ml.evaluation.metrics import KNOWN_LIMITATION_MD_RANGE_M as REGIME_GAP_MD_RANGE_M  # noqa: E402
from ml.features.dataset import load_combined_dataset  # noqa: E402
from ml.features.pipeline import USROPFeatureTransformer  # noqa: E402
from ml.features.split import split_test_cv_pool  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _REPO_ROOT / "docs" / "shap"
RESULTS_PATH = _REPO_ROOT / "docs" / "m5_shap_summary.json"

# The 16-feature M4 candidate this SHAP analysis was computed against (all plots and
# numbers in docs/m5_results.md's global/local/dependence/FR-proxy/latency sections
# describe THIS run). Superseded in MLflow (tag superseded_by) after M5's pipeline
# simplification -- the current m5_candidate=true run is a different LightGBM (14
# features, see docs/m5_simplified_candidate.json) that this script has not been
# re-run against. Update this constant and re-run if a fresh SHAP pass on the current
# candidate is needed.
CANDIDATE_RUN_ID = "cbab6ab7cbbe435da1fc63a9c23a5542"

# REGIME_GAP_MD_RANGE_M is imported above from ml.evaluation.metrics
# (KNOWN_LIMITATION_MD_RANGE_M) -- single source of truth, also used by the backend's
# known_limitation_zone response field (M7). Marked on the MD dependence plot below.


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


def check_fr_regime_proxy(raw: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Is FR (the most important feature) partly a well/regime proxy rather than
    purely physical signal? Computes eta-squared (share of FR's total variance
    explained by well_id, and separately by regime) and saves a boxplot. See
    docs/m5_results.md for the interpretation -- eta-squared, not the ANOVA
    p-value, is what actually answers the question at this sample size.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = raw[["well_id", "FR"]].copy()
    df["regime"] = regime_label(df["well_id"])

    by_well = df.groupby("well_id")["FR"].agg(["mean", "std", "count"])
    by_regime = df.groupby("regime")["FR"].agg(["mean", "std", "count"])

    grand_mean = df["FR"].mean()
    ss_total = float(((df["FR"] - grand_mean) ** 2).sum())
    ss_between_well = float(
        sum(len(g) * (g["FR"].mean() - grand_mean) ** 2 for _, g in df.groupby("well_id"))
    )
    ss_between_regime = float(
        sum(len(g) * (g["FR"].mean() - grand_mean) ** 2 for _, g in df.groupby("regime"))
    )
    eta2_well = ss_between_well / ss_total
    eta2_regime = ss_between_regime / ss_total

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = sorted(df["well_id"].unique())
    data_by_well = [df.loc[df["well_id"] == w, "FR"].to_numpy() for w in order]
    colors = ["#c05621" if w in (0, 1, 6) else "#2b6cb0" for w in order]
    bp = axes[0].boxplot(
        data_by_well, tick_labels=[str(w) for w in order], patch_artist=True, showfliers=False
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xlabel("well_id")
    axes[0].set_ylabel("FR (L/min)")
    axes[0].set_title("FR por pozo (naranja=atipico, azul=dominante)")

    data_by_regime = [df.loc[df["regime"] == r, "FR"].to_numpy() for r in ("dominante", "atipico")]
    bp2 = axes[1].boxplot(
        data_by_regime, tick_labels=["dominante", "atipico"], patch_artist=True, showfliers=False
    )
    bp2["boxes"][0].set_facecolor("#2b6cb0")
    bp2["boxes"][0].set_alpha(0.6)
    bp2["boxes"][1].set_facecolor("#c05621")
    bp2["boxes"][1].set_alpha(0.6)
    axes[1].set_xlabel("regimen")
    axes[1].set_ylabel("FR (L/min)")
    axes[1].set_title("FR por regimen")
    fig.tight_layout()
    fig.savefig(output_dir / "fr_proxy_check.png", dpi=120)
    plt.close(fig)

    return {
        "by_well": by_well.to_dict(orient="index"),
        "by_regime": by_regime.to_dict(orient="index"),
        "eta_squared_well_id": eta2_well,
        "eta_squared_regime": eta2_regime,
    }


def profile_local_explanation_latency(
    explainer: shap.TreeExplainer, model: Any, X: pd.DataFrame, n_samples: int = 100, seed: int = 42
) -> dict[str, float]:
    """p50/p95/p99 latency of explaining ONE row at a time (as a real inference
    request would), plus the same for a plain model.predict, for comparison. Excludes
    a warm-up call from the timing.
    """
    import time

    _ = explainer(X.iloc[[0]])
    _ = model.predict(X.iloc[[0]])

    rng = np.random.default_rng(seed)
    positions = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    explain_latencies_ms = []
    predict_latencies_ms = []
    for pos in positions:
        row = X.iloc[[pos]]
        t0 = time.perf_counter()
        _ = explainer(row)
        t1 = time.perf_counter()
        explain_latencies_ms.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        _ = model.predict(row)
        t1 = time.perf_counter()
        predict_latencies_ms.append((t1 - t0) * 1000)

    explain_arr = np.array(explain_latencies_ms)
    predict_arr = np.array(predict_latencies_ms)
    return {
        "n": len(explain_arr),
        "explain_p50_ms": float(np.percentile(explain_arr, 50)),
        "explain_p95_ms": float(np.percentile(explain_arr, 95)),
        "explain_p99_ms": float(np.percentile(explain_arr, 99)),
        "explain_mean_ms": float(explain_arr.mean()),
        "predict_only_p50_ms": float(np.percentile(predict_arr, 50)),
        "predict_only_p95_ms": float(np.percentile(predict_arr, 95)),
        "predict_only_mean_ms": float(predict_arr.mean()),
    }


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

    logger.info("Chequeando si FR es proxy de pozo/regimen...")
    fr_proxy_results = check_fr_regime_proxy(raw_full, OUTPUT_DIR)
    logger.info(
        "eta^2(well_id)=%.4f eta^2(regime)=%.4f",
        fr_proxy_results["eta_squared_well_id"],
        fr_proxy_results["eta_squared_regime"],
    )

    logger.info("Perfilando latencia de SHAP para prediccion individual...")
    _cv_pool_df, test_df = split_test_cv_pool(load_combined_dataset())
    X_test = USROPFeatureTransformer().transform(test_df)
    explainer = shap.TreeExplainer(model)
    latency_results = profile_local_explanation_latency(explainer, model, X_test)
    logger.info(
        "Latencia explain: p50=%.2fms p95=%.2fms (vs predict-only p50=%.2fms)",
        latency_results["explain_p50_ms"],
        latency_results["explain_p95_ms"],
        latency_results["predict_only_p50_ms"],
    )

    results = {
        "candidate_run_id": CANDIDATE_RUN_ID,
        "n_rows_explained": len(X_full),
        "additivity_max_error": additivity_error,
        "mean_abs_shap_by_feature": ranked,
        "fr_regime_proxy_check": fr_proxy_results,
        "local_explanation_latency": latency_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultados guardados en %s", RESULTS_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
