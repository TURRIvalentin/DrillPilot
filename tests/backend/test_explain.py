"""Tests for POST /explain -- prediction + SHAP breakdown of the last reading in the
window (see docs/adr/005-shap-endpoint-design.md)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient


def test_explain_returns_contribution_per_feature(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0, 150.0, 200.0])

    response = explain_client.post("/explain", json={"readings": readings})

    assert response.status_code == 200
    body = response.json()
    assert body["md"] == 200.0
    assert len(body["contributions"]) == 14  # 12 direct + 2 rolling-window features
    assert {c["feature"] for c in body["contributions"]} >= {"MD", "WOB", "RPM"}


def test_explain_satisfies_shap_additivity(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0, 150.0, 200.0])

    response = explain_client.post("/explain", json={"readings": readings})

    body = response.json()
    reconstructed = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert reconstructed == pytest.approx(body["predicted_rop"], abs=1e-6)


def test_explain_explains_only_the_last_reading(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0, 150.0, 999.0])

    response = explain_client.post("/explain", json={"readings": readings})

    # per ADR-005, /explain always describes the last row of the window, not the
    # whole window -- md in the response must match the last reading sent.
    assert response.json()["md"] == 999.0


def test_explain_rejects_empty_readings_list(explain_client: TestClient) -> None:
    response = explain_client.post("/explain", json={"readings": []})

    assert response.status_code == 422


def test_explain_returns_503_when_model_not_loaded(
    unloaded_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0])

    response = unloaded_client.post("/explain", json={"readings": readings})

    assert response.status_code == 503
