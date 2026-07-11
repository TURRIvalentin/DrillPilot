"""Tests for ml.inference.export_model.save_portable_artifact -- the round-trip
this depends on (mlflow.sklearn.save_model -> load_model from a bare local path,
no tracking store) is the core mechanism docs/adr/006-model-packaging-deploy.md
relies on for baking the model into the Docker image. load_pinned_run_model /
export_model_artifact are not tested here: they need a real local MLflow tracking
store (mlruns/ + the tracking db) populated by M6's promote_model.py, which is a
developer-run precondition, not something the automated suite should depend on --
consistent with how tests/ml/test_shap_explain.py avoids running against the real
registry too.
"""

from __future__ import annotations

from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.features.pipeline import USROPFeatureTransformer
from ml.inference.export_model import save_portable_artifact
from ml.models.dummy_baseline import GlobalMeanBaseline


def _history_frame(n: int = 5, well_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [well_id] * n,
            "MD": [float(i) for i in range(n)],
            "WOB": [5.0] * n,
            "SPP": [1000.0] * n,
            "T": [1.0] * n,
            "RPM": [100.0] * n,
            "FR": [2000.0] * n,
            "DS": [1.2] * n,
            "HD": [300.0] * n,
            "HL": [90.0] * n,
            "VD": [float(i) for i in range(n)],
            "GR": [50.0] * n,
            "gr_imputed": [False] * n,
        }
    )


def _fitted_pipeline() -> Pipeline:
    history = _history_frame()
    model = GlobalMeanBaseline().fit(history, pd.Series([20.0] * len(history)))
    return Pipeline([("features", USROPFeatureTransformer()), ("model", model)])


def test_save_portable_artifact_round_trips_via_a_bare_local_path(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    out_dir = tmp_path / "exported"

    returned = save_portable_artifact(pipeline, out_dir)

    assert returned == out_dir
    assert (out_dir / "MLmodel").exists()

    reloaded = mlflow.sklearn.load_model(str(out_dir))
    history = _history_frame(3)
    assert list(reloaded.predict(history)) == list(pipeline.predict(history))


def test_save_portable_artifact_overwrites_an_existing_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "exported"
    out_dir.mkdir()
    (out_dir / "stale_file.txt").write_text("leftover from a previous export")

    save_portable_artifact(_fitted_pipeline(), out_dir)

    assert not (out_dir / "stale_file.txt").exists()
    assert (out_dir / "MLmodel").exists()
