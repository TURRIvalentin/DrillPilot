"""End-to-end tests for the Streamlit page (frontend/streamlit_app/app.py), driven
via streamlit.testing.v1.AppTest -- runs the real script, no live backend needed:
api_client.predict/explain are monkeypatched at the module the app calls them
through, same principle as tests/backend/conftest.py injecting a fake model instead
of hitting MLflow.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from frontend.streamlit_app import api_client
from frontend.streamlit_app.formatting import (
    INSUFFICIENT_HISTORY_MESSAGE,
    KNOWN_LIMITATION_ZONE_MESSAGE,
)

APP_PATH = "frontend/streamlit_app/app.py"


def _run_app(patched_dataset: pd.DataFrame) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    # The first AppTest run in a process pays a one-time Streamlit runtime/cache
    # cold-start cost that exceeds the 3s default -- later runs are well under it.
    at.run(timeout=20)
    assert not at.exception
    return at


def test_app_loads_without_error_and_shows_a_sample_window(patched_dataset: pd.DataFrame) -> None:
    at = _run_app(patched_dataset)

    assert "DrillPilot" in at.title[0].value
    assert not at.session_state["readings_df"].empty


def test_loading_a_different_sample_replaces_the_window(patched_dataset: pd.DataFrame) -> None:
    at = _run_app(patched_dataset)

    at.selectbox[0].select(5).run()
    load_button = next(b for b in at.button if b.label == "Cargar ejemplo real")
    load_button.click().run()

    assert not at.exception
    assert (at.session_state["readings_df"]["well_id"] == 5).all()


def test_predict_shows_known_limitation_and_insufficient_history_warnings(
    monkeypatch: pytest.MonkeyPatch, patched_dataset: pd.DataFrame
) -> None:
    def fake_predict(base_url: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "predictions": [
                {"md": 700.0, "predicted_rop": 10.0, "known_limitation_zone": True},
            ],
            "insufficient_history": True,
        }

    monkeypatch.setattr(api_client, "predict", fake_predict)

    at = _run_app(patched_dataset)
    predict_button = next(b for b in at.button if b.label == "Predecir")
    predict_button.click().run()

    assert not at.exception
    warning_texts = [w.value for w in at.warning]
    assert KNOWN_LIMITATION_ZONE_MESSAGE in warning_texts
    assert INSUFFICIENT_HISTORY_MESSAGE in warning_texts


def test_predict_shows_no_warnings_when_both_flags_are_false(
    monkeypatch: pytest.MonkeyPatch, patched_dataset: pd.DataFrame
) -> None:
    def fake_predict(base_url: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "predictions": [
                {"md": 100.0, "predicted_rop": 10.0, "known_limitation_zone": False},
            ],
            "insufficient_history": False,
        }

    monkeypatch.setattr(api_client, "predict", fake_predict)

    at = _run_app(patched_dataset)
    predict_button = next(b for b in at.button if b.label == "Predecir")
    predict_button.click().run()

    assert not at.exception
    assert len(at.warning) == 0


def test_predict_surfaces_backend_error(
    monkeypatch: pytest.MonkeyPatch, patched_dataset: pd.DataFrame
) -> None:
    def fake_predict(base_url: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
        raise api_client.BackendError("/predict -> HTTP 422: MD no es no-decreciente")

    monkeypatch.setattr(api_client, "predict", fake_predict)

    at = _run_app(patched_dataset)
    predict_button = next(b for b in at.button if b.label == "Predecir")
    predict_button.click().run()

    assert not at.exception
    assert any("MD no es no-decreciente" in e.value for e in at.error)


def test_explain_shows_warnings_and_prediction(
    monkeypatch: pytest.MonkeyPatch, patched_dataset: pd.DataFrame
) -> None:
    def fake_explain(base_url: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "md": 700.0,
            "predicted_rop": 12.5,
            "known_limitation_zone": True,
            "insufficient_history": False,
            "base_value": 20.0,
            "contributions": [
                {"feature": "MD", "value": 700.0, "shap_value": -5.0},
                {"feature": "WOB", "value": 5.0, "shap_value": -2.5},
            ],
        }

    monkeypatch.setattr(api_client, "explain", fake_explain)

    at = _run_app(patched_dataset)
    explain_button = next(b for b in at.button if b.label == "Explicar (SHAP)")
    explain_button.click().run()

    assert not at.exception
    assert [w.value for w in at.warning] == [KNOWN_LIMITATION_ZONE_MESSAGE]
    assert at.metric[0].value == "12.50"
