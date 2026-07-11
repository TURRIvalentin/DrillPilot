"""M8: thin HTTP client for the DrillPilot backend (M7 FastAPI -- /predict,
/explain, /health). Pure functions, no Streamlit dependency, so they're testable
without a running app -- app.py is the only module in this package that imports
streamlit itself.

Never reimplements request/response validation: whatever the backend's Pydantic
schemas accept or reject is exactly what these functions send and return, as plain
dicts -- see backend/app/schemas/ for the authoritative shape.
"""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_TIMEOUT_S = 10.0


class BackendError(RuntimeError):
    """Raised when the backend responds with a non-2xx status. Carries the
    response body's `detail` message (see backend/app/core/exceptions.py) when the
    backend sent one, otherwise the raw response text."""


def _extract_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text
    detail = body.get("detail") if isinstance(body, dict) else None
    return str(detail) if detail is not None else response.text


def _post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=DEFAULT_TIMEOUT_S)
    if not response.ok:
        raise BackendError(f"{path} -> HTTP {response.status_code}: {_extract_detail(response)}")
    result: dict[str, Any] = response.json()
    return result


def predict(base_url: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
    """POST /predict. `readings` matches backend.app.schemas.reading.Reading
    field-for-field (well_id, MD, WOB, ... gr_imputed). Returns the parsed
    PredictionResponse body: {"predictions": [...], "insufficient_history": bool}.
    """
    return _post(base_url, "/predict", {"readings": readings})


def explain(base_url: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
    """POST /explain -- same request shape as predict(). Returns the parsed
    ExplainResponse body (explains only the last reading, see ADR-005)."""
    return _post(base_url, "/explain", {"readings": readings})


def health(base_url: str) -> dict[str, Any]:
    """GET /health. Raises requests.HTTPError on a non-2xx status (health has no
    request body to validate, so there is no BackendError-style `detail` to surface)."""
    response = requests.get(f"{base_url}/health", timeout=DEFAULT_TIMEOUT_S)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result
