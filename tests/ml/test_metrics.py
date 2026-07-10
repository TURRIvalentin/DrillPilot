"""Tests for the pooled/by-well/by-regime MAE reporting (ml.evaluation.metrics)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.evaluation.metrics import (
    ATYPICAL_REGIME_WELL_IDS,
    DOMINANT_REGIME_WELL_IDS,
    mae_report,
    regime_of,
)


def test_regime_of_classifies_dominant_wells() -> None:
    for well in DOMINANT_REGIME_WELL_IDS:
        assert regime_of(well) == "dominante"


def test_regime_of_classifies_atypical_wells() -> None:
    for well in ATYPICAL_REGIME_WELL_IDS:
        assert regime_of(well) == "atipico"


def test_regime_of_raises_on_unknown_well() -> None:
    with pytest.raises(ValueError):
        regime_of(99)


def test_mae_report_pooled_by_well_and_by_regime() -> None:
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0])
    y_pred = [12.0, 18.0, 33.0, 35.0]
    well_id = pd.Series([2, 2, 0, 0])  # well 2 = dominante, well 0 = atipico

    report = mae_report(y_true, y_pred, well_id)

    assert report.pooled == pytest.approx((2 + 2 + 3 + 5) / 4)
    assert report.by_well[2] == pytest.approx((2 + 2) / 2)
    assert report.by_well[0] == pytest.approx((3 + 5) / 2)
    assert report.by_regime["dominante"] == pytest.approx((2 + 2) / 2)
    assert report.by_regime["atipico"] == pytest.approx((3 + 5) / 2)


def test_to_flat_metrics_naming() -> None:
    y_true = pd.Series([10.0, 20.0])
    y_pred = [10.0, 20.0]
    well_id = pd.Series([3, 3])

    flat = mae_report(y_true, y_pred, well_id).to_flat_metrics("test")

    assert flat["test_pooled"] == 0.0
    assert flat["test_well_3"] == 0.0
    assert flat["test_regime_dominante"] == 0.0
    assert "test_regime_atipico" not in flat
