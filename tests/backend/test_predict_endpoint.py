"""Tests for POST /predict -- the fast, no-SHAP path (see
docs/adr/005-shap-endpoint-design.md)."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient


def test_predict_returns_one_prediction_per_reading(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0, 150.0, 200.0])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert [p["md"] for p in predictions] == [100.0, 150.0, 200.0]
    # dummy_pipeline is GlobalMeanBaseline fit on y=20.0 everywhere -- every
    # prediction should be exactly 20.0 regardless of input.
    assert all(p["predicted_rop"] == 20.0 for p in predictions)


def test_predict_accepts_a_single_reading(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predict_rejects_empty_readings_list(client: TestClient) -> None:
    response = client.post("/predict", json={"readings": []})

    assert response.status_code == 422


def test_predict_rejects_missing_required_field(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0])
    del readings[0]["HD"]

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 422


def test_predict_rejects_non_monotonic_md_within_a_well(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0, 50.0])  # descending MD, same well_id

    response = client.post("/predict", json={"readings": readings})

    # ml.features.pipeline.USROPFeatureTransformer raises ValueError on this, mapped
    # to 422 by backend/app/core/exceptions.py (client error, not a server bug).
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(
    unloaded_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0])

    response = unloaded_client.post("/predict", json={"readings": readings})

    assert response.status_code == 503
