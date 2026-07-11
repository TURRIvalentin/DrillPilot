"""Shared fixtures for frontend tests. No real dataset, no network -- AppTest-based
tests patch ml.features.dataset.load_combined_dataset with a small synthetic frame
(same field-dict pattern as tests/backend/conftest.py) so they never depend on the
real downloaded USROP CSVs being present, consistent with tests/ml/test_features.py's
own use of a tmp_path dataset instead of the real one.
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st


def _synthetic_combined_dataset() -> pd.DataFrame:
    records = [
        {
            "well_id": well_id,
            "MD": 100.0 + i,
            "WOB": 5.0,
            "SPP": 1000.0,
            "T": 1.0,
            "RPM": 100.0,
            "FR": 2000.0,
            "DS": 1.2,
            "HD": 300.0,
            "HL": 90.0,
            "VD": 100.0 + i,
            "GR": 50.0,
            "gr_imputed": False,
        }
        for well_id in range(7)
        for i in range(15)
    ]
    return pd.DataFrame(records)


@pytest.fixture
def patched_dataset(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Patches the dataset loader at its source module -- _cached_dataset() in
    app.py does `from ml.features.dataset import load_combined_dataset` at call
    time (not at app.py's module-load time), so this is observed correctly even
    though AppTest re-executes app.py's script body on every .run().
    st.cache_data.clear() avoids a stale cache entry from a previous test polluting
    this one (cache_data's key does not depend on which mock was active)."""
    df = _synthetic_combined_dataset()
    monkeypatch.setattr("ml.features.dataset.load_combined_dataset", lambda *a, **kw: df)
    st.cache_data.clear()
    return df
