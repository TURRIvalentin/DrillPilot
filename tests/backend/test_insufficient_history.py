"""Dedicated test for the insufficient_history confidence flag: confirms it
activates exactly for windows shorter than ml.features.pipeline.DEFAULT_ROLLING_WINDOW
(10 readings), on both /predict and /explain, analogous to
tests/backend/test_known_limitation_zone.py for known_limitation_zone."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from ml.features.pipeline import DEFAULT_ROLLING_WINDOW

assert DEFAULT_ROLLING_WINDOW == 10  # the values below assume this exact threshold


def test_predict_flags_windows_shorter_than_the_threshold(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    for n_rows in (1, 5, 9):
        readings = make_readings([float(i) for i in range(n_rows)])

        response = client.post("/predict", json={"readings": readings})

        assert response.status_code == 200
        assert response.json()["insufficient_history"] is True, f"n_rows={n_rows}"


def test_predict_does_not_flag_windows_at_or_above_the_threshold(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    for n_rows in (10, 15):
        readings = make_readings([float(i) for i in range(n_rows)])

        response = client.post("/predict", json={"readings": readings})

        assert response.status_code == 200
        assert response.json()["insufficient_history"] is False, f"n_rows={n_rows}"


def test_predict_does_not_reject_a_short_window(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    """ADR-004: a window of 1 row is still valid, just degraded -- insufficient_history
    exposes that, it never rejects the request."""
    readings = make_readings([100.0])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_explain_flags_windows_shorter_than_the_threshold(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([float(i) for i in range(3)])

    response = explain_client.post("/explain", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["insufficient_history"] is True


def test_explain_does_not_flag_windows_at_or_above_the_threshold(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([float(i) for i in range(12)])

    response = explain_client.post("/explain", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["insufficient_history"] is False
