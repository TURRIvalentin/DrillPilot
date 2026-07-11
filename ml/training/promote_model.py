"""M6: build the single serializable artifact (features + model) and promote it in
the MLflow Model Registry -- no feature logic reimplemented in the backend, per the
project's original architecture.

Wraps the M5 candidate LightGBM (tagged m5_candidate=true in the drillpilot-m4-baselines
experiment) together with USROPFeatureTransformer into one sklearn Pipeline, logs it as
a new run/artifact, registers it, and aliases the new version "production".

Note on terminology: MLflow's classic Stage-based registry ("Staging"/"Production") is
deprecated as of MLflow 2.9+ in favor of aliases + tags. This uses
`set_registered_model_alias(..., "production", ...)`, which is the current mechanism
for "this is the version that should be served" -- the same intent, current API.

Run: python -m ml.training.promote_model
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from ml.features.pipeline import USROPFeatureTransformer
from ml.training.train_baselines import EXPERIMENT_NAME

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "drillpilot-rop"
PRODUCTION_ALIAS = "production"


def find_m5_candidate_run_id() -> str:
    """Find the run tagged m5_candidate=true in the M4/M5 experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento MLflow '{EXPERIMENT_NAME}'.")
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.m5_candidate = 'true'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "No hay ninguna corrida tageada m5_candidate=true en MLflow -- "
            "confirmar el candidato (ver docs/m5_results.md) antes de promoverlo."
        )
    return str(runs[0].info.run_id)


def build_combined_pipeline(candidate_run_id: str) -> Pipeline:
    """Wrap the already-tuned/fitted LightGBM together with the feature transformer
    into a single sklearn Pipeline -- one artifact, feature logic never reimplemented
    downstream. Never calls .fit(): both steps are already fitted (the transformer is
    stateless by design, see ml/features/pipeline.py), and Pipeline.predict() does not
    require the pipeline itself to have been fit, only its steps.
    """
    model = mlflow.lightgbm.load_model(f"runs:/{candidate_run_id}/model")
    transformer = USROPFeatureTransformer(rolling_window=10)
    return Pipeline([("features", transformer), ("model", model)])


def promote(candidate_run_id: str | None = None) -> str:
    """Log the combined pipeline as a new MLflow run, register it under
    REGISTERED_MODEL_NAME, and alias the new version "production". Returns the new
    registered model version (as a string).
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    candidate_run_id = candidate_run_id or find_m5_candidate_run_id()
    logger.info(
        "Empaquetando candidato %s (features + modelo, un solo artefacto)...", candidate_run_id
    )
    combined = build_combined_pipeline(candidate_run_id)

    with mlflow.start_run(run_name="promote-to-production") as run:
        mlflow.set_tag("model_family", "combined_pipeline")
        mlflow.set_tag("source_candidate_run_id", candidate_run_id)
        mlflow.sklearn.log_model(
            combined,
            "model",
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        new_run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{new_run_id}'")
    if not versions:
        raise RuntimeError("No se encontro la version recien registrada.")
    version = versions[0].version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, "source_candidate_run_id", candidate_run_id
    )
    logger.info(
        "Promovido: %s version %s -> alias '%s' (candidato origen: %s)",
        REGISTERED_MODEL_NAME,
        version,
        PRODUCTION_ALIAS,
        candidate_run_id,
    )
    return str(version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    promote()
