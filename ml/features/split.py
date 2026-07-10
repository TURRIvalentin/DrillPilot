"""Split the combined USROP dataset into the CV-tuning pool and the final test set,
per docs/adr/003-split-strategy.md. Splits by well_id only -- never by row or index.

Test wells (0, 3, 5) are never touched until terminal evaluation. CV-pool wells
(1, 2, 4, 6) are used for leave-one-well-out cross-validation during tuning.
"""

from __future__ import annotations

import pandas as pd

TEST_WELL_IDS: frozenset[int] = frozenset({0, 3, 5})
CV_POOL_WELL_IDS: frozenset[int] = frozenset({1, 2, 4, 6})


def split_test_cv_pool(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` (must have a well_id column) into (cv_pool_df, test_df) per ADR-003.

    Raises ValueError if `df` contains well_id values outside the 7 known USROP wells.
    """
    if "well_id" not in df.columns:
        raise ValueError("df necesita una columna 'well_id' (ver ml.features.dataset).")

    present = set(df["well_id"].unique())
    known = TEST_WELL_IDS | CV_POOL_WELL_IDS
    unknown = present - known
    if unknown:
        raise ValueError(f"well_id desconocidos para USROP (ADR-003): {sorted(unknown)}")

    cv_pool_df = df[df["well_id"].isin(CV_POOL_WELL_IDS)].reset_index(drop=True)
    test_df = df[df["well_id"].isin(TEST_WELL_IDS)].reset_index(drop=True)
    return cv_pool_df, test_df
