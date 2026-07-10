"""Smoke tests que validan que la estructura de paquetes de M0 es importable."""

import importlib


def test_ml_package_is_importable() -> None:
    assert importlib.import_module("ml") is not None


def test_backend_package_is_importable() -> None:
    assert importlib.import_module("backend") is not None
