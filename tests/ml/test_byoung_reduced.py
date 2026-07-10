"""Tests for the reduced Bourgoyne & Young baseline (4/8 terms)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.byoung_reduced import BourgoyneYoungReduced


def _synthetic_df(
    n_per_well: int = 30, n_wells: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    frames = []
    for well in range(n_wells):
        md = np.linspace(1000, 2000, n_per_well) + well * 100
        wob = rng.uniform(2, 15, n_per_well)
        hd = np.full(n_per_well, 300.0)
        rpm = rng.uniform(80, 200, n_per_well)
        frames.append(pd.DataFrame({"well_id": well, "MD": md, "WOB": wob, "HD": hd, "RPM": rpm}))
    df = pd.concat(frames, ignore_index=True)
    rop = 10 + 0.5 * df["WOB"] + 0.05 * df["RPM"] + rng.normal(0, 1, len(df))
    return df, pd.Series(rop.clip(lower=1.0))


def test_fit_predict_no_nan_or_inf() -> None:
    df, y = _synthetic_df()

    model = BourgoyneYoungReduced().fit(df, y)
    preds = model.predict(df)

    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert (preds > 0).all()


def test_fit_selects_threshold_from_configured_grid() -> None:
    df, y = _synthetic_df()
    grid = (0.0, 0.5)

    model = BourgoyneYoungReduced(wob_threshold_grid=grid).fit(df, y)

    assert model.wob_threshold_ in grid


def test_handles_rpm_zero_rows_without_nan_or_inf() -> None:
    """Regression test: ln(RPM/60) blows up to -inf at RPM==0 (sliding drilling,
    a real and legitimate event per M2 -- see the _RPM_EPSILON docstring). This
    broke the very first real-data run of this model; must never regress."""
    df, y = _synthetic_df()
    df = df.copy()
    df.loc[0, "RPM"] = 0.0

    model = BourgoyneYoungReduced().fit(df, y)
    preds = model.predict(df)

    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert model.rpm_clipped_fraction_ > 0


def test_missing_required_column_raises() -> None:
    df, y = _synthetic_df()
    df = df.drop(columns=["HD"])

    with pytest.raises(ValueError):
        BourgoyneYoungReduced().fit(df, y)


def test_clipped_fraction_diagnostics_are_valid_proportions() -> None:
    df, y = _synthetic_df()

    model = BourgoyneYoungReduced().fit(df, y)

    assert 0.0 <= model.clipped_fraction_ <= 1.0
    assert 0.0 <= model.rpm_clipped_fraction_ <= 1.0


def test_fit_stores_named_coefficients() -> None:
    df, y = _synthetic_df()

    model = BourgoyneYoungReduced().fit(df, y)

    assert isinstance(model.a1_, float)
    assert isinstance(model.a2_, float)
    assert isinstance(model.a5_, float)
    assert isinstance(model.a6_, float)
