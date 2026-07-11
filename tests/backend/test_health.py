"""Tests for GET /health."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_ok_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "model_uri" in body


def test_health_reports_degraded_when_model_not_loaded(unloaded_client: TestClient) -> None:
    response = unloaded_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
