"""M8: loads a real window of USROP readings for the demo's "cargar ejemplo real"
button. This reads local training data to populate the input form -- it does not
touch feature engineering or the model, so it does not duplicate anything the
backend/pipeline already owns (see ml/features/pipeline.py, never imported here).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.features.pipeline import DIRECT_FEATURE_COLUMNS

READING_COLUMNS: tuple[str, ...] = ("well_id", *DIRECT_FEATURE_COLUMNS)

AVAILABLE_WELL_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


def dataframe_to_readings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a readings table (e.g. from st.data_editor, or a sample window) into
    the list-of-dicts shape POST /predict and /explain expect -- coerces well_id to
    int and gr_imputed to bool, since a data editor round-trip can leave these as
    float/numpy types that Reading's Pydantic validation would otherwise reject.
    """
    records: list[dict[str, Any]] = df[list(READING_COLUMNS)].to_dict(orient="records")
    for record in records:
        record["well_id"] = int(record["well_id"])
        record["gr_imputed"] = bool(record["gr_imputed"])
    return records


def load_sample_window(
    df: pd.DataFrame, well_id: int, start_row: int, n_rows: int
) -> list[dict[str, Any]]:
    """`n_rows` consecutive readings from `well_id`, sorted by MD ascending, starting
    at the `start_row`-th row of that well -- the shape POST /predict and /explain
    expect (see backend/app/schemas/reading.py). `df` is the caller's already-loaded
    combined dataset (ml.features.dataset.load_combined_dataset()), passed in so this
    function stays fast and testable with a small synthetic frame.
    """
    well = df[df["well_id"] == well_id].sort_values("MD").reset_index(drop=True)
    window = well.iloc[start_row : start_row + n_rows]
    return dataframe_to_readings(window)
