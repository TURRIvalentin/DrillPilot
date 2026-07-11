"""Tests for frontend.streamlit_app.api_client -- no real network, requests.post/get
are monkeypatched with a small fake response object."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from frontend.streamlit_app.api_client import BackendError, explain, health, predict


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}")


def test_predict_posts_readings_and_returns_parsed_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: Any, timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(200, {"predictions": [], "insufficient_history": False})

    monkeypatch.setattr(requests, "post", fake_post)

    result = predict("http://localhost:8000", [{"well_id": 0, "MD": 100.0}])

    assert captured["url"] == "http://localhost:8000/predict"
    assert captured["json"] == {"readings": [{"well_id": 0, "MD": 100.0}]}
    assert result == {"predictions": [], "insufficient_history": False}


def test_explain_posts_to_the_explain_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: Any, timeout: float) -> _FakeResponse:
        captured["url"] = url
        return _FakeResponse(200, {"md": 100.0})

    monkeypatch.setattr(requests, "post", fake_post)

    result = explain("http://localhost:8000", [{"well_id": 0, "MD": 100.0}])

    assert captured["url"] == "http://localhost:8000/explain"
    assert result == {"md": 100.0}


def test_predict_raises_backend_error_with_detail_on_non_2xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse(422, {"detail": "MD no es no-decreciente"})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(BackendError, match="MD no es no-decreciente"):
        predict("http://localhost:8000", [{"well_id": 0, "MD": 100.0}])


def test_predict_falls_back_to_raw_text_when_body_is_not_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse(500, None, text="internal server error")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(BackendError, match="internal server error"):
        predict("http://localhost:8000", [{"well_id": 0, "MD": 100.0}])


def test_health_calls_get_and_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_ok(url: str, timeout: float) -> _FakeResponse:
        return _FakeResponse(200, {"status": "ok", "model_loaded": True})

    monkeypatch.setattr(requests, "get", fake_get_ok)
    assert health("http://localhost:8000") == {"status": "ok", "model_loaded": True}

    def fake_get_down(url: str, timeout: float) -> _FakeResponse:
        return _FakeResponse(503, {"detail": "unavailable"})

    monkeypatch.setattr(requests, "get", fake_get_down)
    with pytest.raises(requests.HTTPError):
        health("http://localhost:8000")
