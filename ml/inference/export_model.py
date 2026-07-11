"""M9: exports the pinned production model artifact to a plain, portable local
directory the Docker build copies verbatim -- see
docs/adr/006-model-packaging-deploy.md for the full reasoning, including why a
naive `COPY` of MLflow's own internal artifact directory does not work.

This is a developer-run, one-off script -- it needs the local MLflow tracking
store (mlruns/ + the tracking db) to resolve `runs:/{run_id}/model`, so it is
never invoked from inside `docker build` or the running container, only locally
before committing the exported directory.

Run: python -m ml.inference.export_model [--run-id RUN_ID] [--out OUT_DIR]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import mlflow.sklearn

# The run_id of the "promote-to-production" run (ml/training/promote_model.py, M6)
# that registered drillpilot-rop v1 and aliased it "production". Pin this to
# whatever run should ship next -- never resolve the live "production" alias here,
# see ADR-006 point 4.
PINNED_PRODUCTION_RUN_ID = "a302711c1cd34d4da4c86235387dc5f8"

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "docker" / "model_artifact"


def load_pinned_run_model(run_id: str) -> Any:
    """Loads the combined features+model Pipeline from `runs:/{run_id}/model`.
    Requires the local MLflow tracking store to be populated (mlruns/ + the
    tracking db) -- this is the one place in the export flow that touches it."""
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def save_portable_artifact(model: Any, out_dir: Path) -> Path:
    """Re-saves `model` to `out_dir` as a plain, self-contained MLflow model
    directory that `mlflow.sklearn.load_model(out_dir)` can load from a bare local
    path with zero tracking-store dependency -- verified in
    tests/ml/test_export_model.py and docs/m9_m10_results.md (a straight `COPY` of
    MLflow's own internal artifact layout does not have this property on Windows).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    mlflow.sklearn.save_model(model, str(out_dir), serialization_format="cloudpickle")
    return out_dir


def export_model_artifact(run_id: str, out_dir: Path) -> Path:
    model = load_pinned_run_model(run_id)
    return save_portable_artifact(model, out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=PINNED_PRODUCTION_RUN_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = export_model_artifact(args.run_id, Path(args.out))
    print(f"Modelo exportado a {out_dir} (run_id={args.run_id})")


if __name__ == "__main__":
    main()
