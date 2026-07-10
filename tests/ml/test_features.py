"""Tests for the M3 feature pipeline: dataset loading, ADR-003 split, and the
sklearn-compatible window-feature transformer -- with an explicit focus on the two
non-negotiable rules from ADR-003 §3: no leakage across well boundaries, and no
look-ahead in time/depth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.features.dataset import load_combined_dataset, parse_well_id
from ml.features.pipeline import USROPFeatureTransformer
from ml.features.split import CV_POOL_WELL_IDS, TEST_WELL_IDS, split_test_cv_pool

# ---------------------------------------------------------------------------
# ml.features.dataset
# ---------------------------------------------------------------------------

REAL_USROP_FILENAMES = [
    "USROP_A 0 N-NA_F-9_Ad.csv",
    "USROP_A 1 N-S_F-7d.csv",
    "USROP_A 2 N-SH_F-14d.csv",
    "USROP_A 3 N-SH-F-15d.csv",
    "USROP_A 4 N-SH_F-15Sd.csv",
    "USROP_A 5 N-SH-F-5d.csv",
    "USROP_A 6 N-SH_F-9d.csv",
]


@pytest.mark.parametrize(
    ("filename", "expected"), list(zip(REAL_USROP_FILENAMES, range(7), strict=False))
)
def test_parse_well_id_matches_known_filenames(filename: str, expected: int) -> None:
    assert parse_well_id(filename) == expected


def test_parse_well_id_raises_on_unexpected_format() -> None:
    with pytest.raises(ValueError):
        parse_well_id("not_a_usrop_file.csv")


_RAW_HEADER = (
    "Measured Depth m,Weight on Bit kkgf,Average Standpipe Pressure kPa,"
    "Average Surface Torque kN.m,Rate of Penetration m/h,Average Rotary Speed rpm,"
    "Mud Flow In L/min,Mud Density In g/cm3,Diameter mm,Average Hookload kkgf,"
    "Hole Depth (TVD) m,USROP Gamma gAPI"
)


def _write_raw_csv(path: Path, rows: list[tuple[float, ...]]) -> None:
    lines = [f",{_RAW_HEADER}"]
    for i, row in enumerate(rows):
        lines.append(f"{i}," + ",".join(str(v) for v in row))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_combined_dataset_tags_well_id_and_applies_cleaning(tmp_path: Path) -> None:
    # MD, WOB, SPP, T, ROP, RPM, FR, DS, HD, HL, VD, GR
    well0_rows = [
        (100.0, 5.0, 1000.0, 1.0, 20.0, 100.0, 2000.0, 1.2, 300.0, 90.0, 100.0, 10.0),
        (100.1, 5.0, 1000.0, 1.0, 20.0, 100.0, 2000.0, 1.2, 300.0, 90.0, 100.1, 0.0),
        (100.2, 5.0, 1000.0, 1.0, 20.0, 100.0, 2000.0, 1.2, 300.0, 90.0, 100.2, 30.0),
    ]
    well3_rows = [
        (200.0, 6.0, 1100.0, 1.5, 25.0, 110.0, 2100.0, 1.3, 310.0, 95.0, 200.0, 40.0),
    ]
    _write_raw_csv(tmp_path / "USROP_A 0 N-NA_F-9_Ad.csv", well0_rows)
    _write_raw_csv(tmp_path / "USROP_A 3 N-SH-F-15d.csv", well3_rows)

    combined = load_combined_dataset(tmp_path)

    assert len(combined) == 4
    assert sorted(combined["well_id"].unique().tolist()) == [0, 3]
    assert combined.loc[combined["well_id"] == 0, "GR"].tolist() == [10.0, 20.0, 30.0]
    assert combined.loc[combined["well_id"] == 0, "gr_imputed"].tolist() == [False, True, False]


def test_load_combined_dataset_raises_when_no_csv_present(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_combined_dataset(tmp_path)


# ---------------------------------------------------------------------------
# ml.features.split
# ---------------------------------------------------------------------------


def _tiny_multiwell_df() -> pd.DataFrame:
    return pd.DataFrame({"well_id": list(range(7)), "value": list(range(7))})


def test_split_test_cv_pool_matches_adr003() -> None:
    df = _tiny_multiwell_df()

    cv_pool_df, test_df = split_test_cv_pool(df)

    assert set(test_df["well_id"]) == TEST_WELL_IDS == {0, 3, 5}
    assert set(cv_pool_df["well_id"]) == CV_POOL_WELL_IDS == {1, 2, 4, 6}
    assert len(cv_pool_df) + len(test_df) == len(df)


def test_split_raises_on_unknown_well_id() -> None:
    df = pd.DataFrame({"well_id": [0, 3, 99], "value": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_test_cv_pool(df)


def test_split_raises_when_well_id_column_missing() -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_test_cv_pool(df)


# ---------------------------------------------------------------------------
# ml.features.pipeline.USROPFeatureTransformer
# ---------------------------------------------------------------------------


def _well_frame(well_id: int, md: list[float], wob: list[float], **overrides: list) -> pd.DataFrame:
    n = len(md)
    data = {
        "well_id": [well_id] * n,
        "MD": md,
        "WOB": wob,
        "SPP": overrides.get("SPP", [1000.0] * n),
        "T": overrides.get("T", [1.0] * n),
        "RPM": overrides.get("RPM", [100.0] * n),
        "FR": overrides.get("FR", [2000.0] * n),
        "DS": overrides.get("DS", [1.2] * n),
        "HD": overrides.get("HD", [300.0] * n),
        "HL": overrides.get("HL", [90.0] * n),
        "VD": overrides.get("VD", md),
        "GR": overrides.get("GR", [50.0] * n),
        "gr_imputed": overrides.get("gr_imputed", [False] * n),
    }
    return pd.DataFrame(data)


def test_transform_output_shape_and_columns() -> None:
    df = pd.concat(
        [
            _well_frame(0, md=[1.0, 2.0, 3.0], wob=[10.0, 20.0, 30.0]),
            _well_frame(1, md=[1.0, 2.0], wob=[5.0, 6.0]),
        ],
        ignore_index=True,
    )

    out = USROPFeatureTransformer(rolling_window=2).fit_transform(df)

    assert len(out) == len(df)
    assert list(out.columns) == USROPFeatureTransformer(rolling_window=2).get_feature_names_out()


def test_no_leakage_across_well_boundary() -> None:
    well_a = _well_frame(0, md=[1.0, 2.0, 3.0, 4.0, 5.0], wob=[100.0, 100.0, 100.0, 100.0, 100.0])
    well_b = _well_frame(1, md=[1.0, 2.0, 3.0, 4.0, 5.0], wob=[1.0, 2.0, 3.0, 4.0, 5.0])
    df = pd.concat([well_a, well_b], ignore_index=True)

    out = USROPFeatureTransformer(rolling_window=3).fit_transform(df)

    first_row_well_b = out.iloc[5]
    assert first_row_well_b["WOB_rolling_mean_3"] == 1.0
    assert first_row_well_b["WOB_diff_1"] == 0.0


def test_no_lookahead_truncating_future_rows_does_not_change_past_features() -> None:
    df = _well_frame(0, md=list(range(1, 11)), wob=[float(i) for i in range(1, 11)])
    transformer = USROPFeatureTransformer(rolling_window=3)

    full = transformer.fit_transform(df)
    truncated = transformer.fit_transform(df.iloc[:5].reset_index(drop=True))

    pd.testing.assert_frame_equal(truncated, full.iloc[:5].reset_index(drop=True))


def test_diff_and_rolling_std_first_row_per_well_are_zero_not_nan() -> None:
    df = pd.concat(
        [
            _well_frame(0, md=[1.0, 2.0], wob=[10.0, 20.0], T=[1.0, 2.0]),
            _well_frame(1, md=[1.0, 2.0], wob=[5.0, 6.0], T=[3.0, 4.0]),
        ],
        ignore_index=True,
    )

    out = USROPFeatureTransformer(rolling_window=3).fit_transform(df)

    first_rows = out.iloc[[0, 2]]
    assert (first_rows["WOB_diff_1"] == 0.0).all()
    assert (first_rows["T_rolling_std_3"] == 0.0).all()
    assert not first_rows[["WOB_diff_1", "T_rolling_std_3"]].isna().any().any()


def test_rolling_mean_matches_manual_calculation() -> None:
    df = _well_frame(0, md=[1.0, 2.0, 3.0, 4.0, 5.0], wob=[10.0, 20.0, 30.0, 40.0, 50.0])

    out = USROPFeatureTransformer(rolling_window=3).fit_transform(df)

    assert out["WOB_rolling_mean_3"].tolist() == [10.0, 15.0, 20.0, 30.0, 40.0]


def test_gr_imputed_is_cast_to_int() -> None:
    df = _well_frame(0, md=[1.0, 2.0], wob=[10.0, 20.0], gr_imputed=[True, False])

    out = USROPFeatureTransformer().fit_transform(df)

    assert out["gr_imputed"].tolist() == [1, 0]
    assert out["gr_imputed"].dtype.kind == "i"


def test_validate_input_raises_on_missing_columns() -> None:
    df = _well_frame(0, md=[1.0, 2.0], wob=[10.0, 20.0]).drop(columns=["SPP"])

    with pytest.raises(ValueError):
        USROPFeatureTransformer().fit(df)


def test_validate_input_raises_on_non_monotonic_md_within_well() -> None:
    df = _well_frame(0, md=[10.0, 5.0, 20.0], wob=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        USROPFeatureTransformer().fit(df)
