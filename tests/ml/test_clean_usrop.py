"""Tests for the M2 cleaning rules (docs/cleaning_rules.md), one per rule."""

from __future__ import annotations

import pandas as pd

from ml.cleaning.clean_usrop import apply_cleaning_rules


def _well(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_rule1_rpm_zero_rows_are_left_unchanged() -> None:
    df = _well(
        MD=[100.0, 100.1, 100.2],
        WOB=[5.0, 18.9, 6.0],
        RPM=[120.0, 0.0, 130.0],
        T=[2.0, 2.5, 2.1],
        ROP=[40.0, 31.5, 42.0],
        GR=[60.0, 61.0, 62.0],
    )

    cleaned = apply_cleaning_rules(df)

    assert cleaned.loc[1, ["WOB", "RPM", "T", "ROP", "GR"]].tolist() == [18.9, 0.0, 2.5, 31.5, 61.0]


def test_rule2_low_rop_rows_are_left_unchanged() -> None:
    df = _well(
        MD=[200.0, 200.1, 200.2],
        WOB=[5.0, 6.0, 5.5],
        RPM=[120.0, 118.0, 121.0],
        T=[2.0, 2.1, 2.0],
        ROP=[40.0, 0.3, 41.0],
        GR=[60.0, 61.0, 62.0],
    )

    cleaned = apply_cleaning_rules(df)

    assert cleaned.loc[1, ["WOB", "RPM", "T", "ROP", "GR"]].tolist() == [6.0, 118.0, 2.1, 0.3, 61.0]


def test_rule3_no_rows_are_ever_dropped() -> None:
    df = _well(
        MD=[1.0, 2.0, 3.0, 4.0],
        WOB=[5.0] * 4,
        RPM=[100.0] * 4,
        T=[2.0] * 4,
        ROP=[40.0] * 4,
        GR=[10.0, 0.0, 0.0, 20.0],
    )

    cleaned = apply_cleaning_rules(df)

    assert len(cleaned) == len(df)


def test_rule3_gr_zero_interior_gap_is_linearly_interpolated() -> None:
    df = _well(
        MD=[1.0, 2.0, 3.0, 4.0],
        WOB=[5.0] * 4,
        RPM=[100.0] * 4,
        T=[2.0] * 4,
        ROP=[40.0] * 4,
        GR=[10.0, 0.0, 0.0, 40.0],
    )

    cleaned = apply_cleaning_rules(df)

    assert cleaned["GR"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert cleaned["gr_imputed"].tolist() == [False, True, True, False]


def test_rule3_gr_zero_at_edge_falls_back_to_nearest_valid_value() -> None:
    df = _well(
        MD=[1.0, 2.0, 3.0, 4.0],
        WOB=[5.0] * 4,
        RPM=[100.0] * 4,
        T=[2.0] * 4,
        ROP=[40.0] * 4,
        GR=[0.0, 0.0, 10.0, 20.0],
    )

    cleaned = apply_cleaning_rules(df)

    assert cleaned["GR"].tolist() == [10.0, 10.0, 10.0, 20.0]
    assert cleaned["gr_imputed"].tolist() == [True, True, False, False]


def test_rule3_gr_column_untouched_when_no_zeros_present() -> None:
    df = _well(
        MD=[1.0, 2.0, 3.0],
        WOB=[5.0] * 3,
        RPM=[100.0] * 3,
        T=[2.0] * 3,
        ROP=[40.0] * 3,
        GR=[15.0, 16.0, 17.0],
    )

    cleaned = apply_cleaning_rules(df)

    assert cleaned["GR"].tolist() == [15.0, 16.0, 17.0]
    assert cleaned["gr_imputed"].tolist() == [False, False, False]
