"""Tests for frontend.streamlit_app.sample_data -- uses small synthetic frames, no
real dataset access."""

from __future__ import annotations

import pandas as pd

from frontend.streamlit_app.sample_data import (
    READING_COLUMNS,
    dataframe_to_readings,
    load_sample_window,
)


def _synthetic_well(well_id: int, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [well_id] * n,
            "MD": [float(i) for i in range(n)],
            "WOB": [5.0] * n,
            "SPP": [1000.0] * n,
            "T": [1.0] * n,
            "RPM": [100.0] * n,
            "FR": [2000.0] * n,
            "DS": [1.2] * n,
            "HD": [300.0] * n,
            "HL": [90.0] * n,
            "VD": [float(i) for i in range(n)],
            "GR": [50.0] * n,
            "gr_imputed": [False] * n,
        }
    )


def test_load_sample_window_selects_the_right_well_and_slice() -> None:
    df = pd.concat([_synthetic_well(0, 5), _synthetic_well(3, 20)], ignore_index=True)

    window = load_sample_window(df, well_id=3, start_row=2, n_rows=4)

    assert len(window) == 4
    assert all(r["well_id"] == 3 for r in window)
    assert [r["MD"] for r in window] == [2.0, 3.0, 4.0, 5.0]


def test_load_sample_window_sorts_by_md_ascending() -> None:
    df = _synthetic_well(0, 5).iloc[::-1]  # shuffled/descending input order

    window = load_sample_window(df, well_id=0, start_row=0, n_rows=5)

    assert [r["MD"] for r in window] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_load_sample_window_coerces_well_id_and_gr_imputed_types() -> None:
    df = _synthetic_well(0, 3)

    window = load_sample_window(df, well_id=0, start_row=0, n_rows=3)

    assert all(isinstance(r["well_id"], int) for r in window)
    assert all(isinstance(r["gr_imputed"], bool) for r in window)


def test_load_sample_window_returns_only_reading_columns() -> None:
    df = _synthetic_well(0, 3)
    df["ROP"] = 20.0  # extra column present in the real dataset, not part of Reading

    window = load_sample_window(df, well_id=0, start_row=0, n_rows=3)

    assert set(window[0].keys()) == set(READING_COLUMNS)


def test_dataframe_to_readings_round_trips_an_edited_table() -> None:
    df = _synthetic_well(1, 3)
    df["gr_imputed"] = [1.0, 0.0, 1.0]  # e.g. as a data_editor might leave it

    readings = dataframe_to_readings(df)

    assert readings[0]["gr_imputed"] is True
    assert readings[1]["gr_imputed"] is False
    assert all(isinstance(r["well_id"], int) for r in readings)
