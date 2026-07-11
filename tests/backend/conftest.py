"""Shared fixtures for backend tests. No MLflow registry access, no network -- every
test injects a small, fast, already-fitted pipeline into the app via dependency
override, same pattern as tests/ml/test_predict.py.

/explain needs a real tree model (shap.TreeExplainer does not support
GlobalMeanBaseline), so its pipeline uses a tiny LightGBM instead of the dummy used
for /predict and /health.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import lightgbm as lgb
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from backend.app.api.deps import get_inference_service
from backend.app.main import app
from backend.app.services.inference_service import InferenceService
from ml.evaluation.metrics import KNOWN_LIMITATION_MD_RANGE_M
from ml.features.pipeline import USROPFeatureTransformer
from ml.models.dummy_baseline import GlobalMeanBaseline


def _history_rows(mds: list[float], well_id: int = 0) -> list[dict[str, object]]:
    """One dict per reading with every field Reading requires (see
    backend/app/schemas/reading.py), MD varying per row and everything else constant.
    """
    return [
        {
            "well_id": well_id,
            "MD": md,
            "WOB": 5.0,
            "SPP": 1000.0,
            "T": 1.0,
            "RPM": 100.0,
            "FR": 2000.0,
            "DS": 1.2,
            "HD": 300.0,
            "HL": 90.0,
            "VD": md,
            "GR": 50.0,
            "gr_imputed": False,
        }
        for md in mds
    ]


def _history_frame(mds: list[float], well_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame(_history_rows(mds, well_id))


@pytest.fixture
def dummy_pipeline() -> Pipeline:
    """A fast, non-tree pipeline for /predict and /health -- mirrors
    tests/ml/test_predict.py's fixture."""
    history = _history_frame([float(i) for i in range(5)])
    model = GlobalMeanBaseline().fit(history, pd.Series([20.0] * len(history)))
    return Pipeline([("features", USROPFeatureTransformer()), ("model", model)])


@pytest.fixture
def lightgbm_pipeline() -> Pipeline:
    """A tiny real LightGBM for /explain -- shap.TreeExplainer requires an actual
    tree-based model, GlobalMeanBaseline does not qualify."""
    history = _history_frame([float(i) for i in range(20)])
    y = pd.Series([20.0 + 0.1 * i for i in range(20)])
    model = lgb.LGBMRegressor(n_estimators=5, num_leaves=3, min_child_samples=1, verbose=-1)
    features = USROPFeatureTransformer()
    X = features.fit_transform(history)
    model.fit(X, y)
    return Pipeline([("features", features), ("model", model)])


@pytest.fixture
def client(dummy_pipeline: Pipeline) -> Iterator[TestClient]:
    """TestClient wired to a loaded InferenceService holding the fast dummy pipeline
    -- used for /predict and /health tests, which don't need SHAP."""
    service = InferenceService(model=dummy_pipeline)
    app.dependency_overrides[get_inference_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def explain_client(lightgbm_pipeline: Pipeline) -> Iterator[TestClient]:
    """TestClient wired to a loaded InferenceService holding a real tree model --
    used for /explain tests, which need shap.TreeExplainer to work."""
    service = InferenceService(model=lightgbm_pipeline)
    app.dependency_overrides[get_inference_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def unloaded_client() -> Iterator[TestClient]:
    """TestClient wired to an InferenceService with no model loaded -- used to test
    the 503 ModelNotLoadedError path."""
    service = InferenceService(model=None)
    app.dependency_overrides[get_inference_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def known_limitation_md() -> float:
    """A depth inside the documented high-error band (see
    ml/evaluation/metrics.KNOWN_LIMITATION_MD_RANGE_M)."""
    low, high = KNOWN_LIMITATION_MD_RANGE_M
    return (low + high) / 2


@pytest.fixture
def outside_limitation_md() -> float:
    """A depth clearly outside the documented high-error band."""
    low, _high = KNOWN_LIMITATION_MD_RANGE_M
    return low - 100.0


@pytest.fixture
def make_readings() -> Callable[..., list[dict[str, object]]]:
    """Factory building request-ready reading dicts for arbitrary MD values, with
    every other required field held constant -- see backend/app/schemas/reading.py."""
    return _history_rows
