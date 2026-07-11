"""Shared FastAPI dependencies for the API routers."""

from __future__ import annotations

from fastapi import Request

from backend.app.services.inference_service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """The single InferenceService instance created in main.py's lifespan and
    stored on app.state -- one model load per process, shared by every request."""
    service: InferenceService = request.app.state.inference_service
    return service
