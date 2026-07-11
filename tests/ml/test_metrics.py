"""Tests for the pooled/by-well/by-regime MAE reporting (ml.evaluation.metrics)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.evaluation.metrics import (
    ATYPICAL_REGIME_WELL_IDS,
    DOMINANT_REGIME_WELL_IDS,
    KNOWN_LIMITATION_MD_RANGE_M,
    is_in_known_limitation_zone,
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


def test_known_limitation_md_range_is_the_documented_cv_pool_gap() -> None:
    low, high = KNOWN_LIMITATION_MD_RANGE_M
    assert low < high
    # Matches docs/m4_results.md: atypical CV-pool wells (1, 6) top out at 634 m,
    # dominant CV-pool wells (2, 4) start at 988 m.
    assert low == 634.0
    assert high == 988.0


def test_is_in_known_limitation_zone_boundaries_are_inclusive() -> None:
    assert is_in_known_limitation_zone(634.0) is True
    assert is_in_known_limitation_zone(988.0) is True
    assert is_in_known_limitation_zone(800.0) is True


def test_is_in_known_limitation_zone_outside_the_band() -> None:
    assert is_in_known_limitation_zone(633.9) is False
    assert is_in_known_limitation_zone(988.1) is False
    assert is_in_known_limitation_zone(50.0) is False
