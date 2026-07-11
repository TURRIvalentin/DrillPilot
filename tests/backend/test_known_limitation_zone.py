"""Dedicated test for the known_limitation_zone confidence flag: confirms it
activates exactly for MD inside ml.evaluation.metrics.KNOWN_LIMITATION_MD_RANGE_M
(634-988 m, see docs/m6_results.md), on both /predict and /explain, per the user's
explicit M7 requirement."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from ml.evaluation.metrics import KNOWN_LIMITATION_MD_RANGE_M

_LOW, _HIGH = KNOWN_LIMITATION_MD_RANGE_M


def test_predict_flags_md_inside_the_known_limitation_zone(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([(_LOW + _HIGH) / 2])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["predictions"][0]["known_limitation_zone"] is True


def test_predict_does_not_flag_md_outside_the_known_limitation_zone(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([_LOW - 100.0])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["predictions"][0]["known_limitation_zone"] is False


def test_predict_flags_are_inclusive_on_both_boundaries(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([_LOW, _HIGH])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert predictions[0]["known_limitation_zone"] is True
    assert predictions[1]["known_limitation_zone"] is True


def test_predict_flag_flips_just_past_the_upper_boundary(
    client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([_HIGH + 0.1])

    response = client.post("/predict", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["predictions"][0]["known_limitation_zone"] is False


def test_explain_flags_md_inside_the_known_limitation_zone(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([100.0, (_LOW + _HIGH) / 2])  # ascending MD, last row explained

    response = explain_client.post("/explain", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["known_limitation_zone"] is True


def test_explain_does_not_flag_md_outside_the_known_limitation_zone(
    explain_client: TestClient, make_readings: Callable[..., list[dict[str, object]]]
) -> None:
    readings = make_readings([50.0, 100.0])

    response = explain_client.post("/explain", json={"readings": readings})

    assert response.status_code == 200
    assert response.json()["known_limitation_zone"] is False
