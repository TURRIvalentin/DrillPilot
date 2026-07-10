"""Generate the M2 EDA artifacts: per-well distribution plots and a zero-value
investigation summary, used as evidence for docs/eda_findings.md.

Reads the 7 raw USROP CSVs from data/raw/, writes PNG plots to docs/eda/, and a
JSON summary of the RPM==0 / GR==0 / ROP<1.0 investigation to
docs/eda/zero_value_investigation.json so the numbers cited in
docs/eda_findings.md are reproducible from this script, not hand-typed.

This script does not clean or transform the data -- see docs/cleaning_rules.md
and the (pending, post-approval) cleaning module for that.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ml.data.schema import COLUMN_RENAME  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "docs" / "eda"

DISTRIBUTION_VARIABLES: tuple[str, ...] = ("ROP", "WOB", "RPM", "T", "SPP")


def load_wells(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load each USROP CSV in `raw_data_dir`, renaming columns to their short codes."""
    wells: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_data_dir.glob("*.csv")):
        wells[path.name] = pd.read_csv(path, index_col=0).rename(columns=COLUMN_RENAME)
    if not wells:
        raise FileNotFoundError(
            f"No se encontraron CSV en {raw_data_dir}. Corre ml/data/download_usrop.py primero."
        )
    return wells


def plot_distribution_by_well(
    wells: dict[str, pd.DataFrame], variable: str, output_dir: Path
) -> Path:
    """Save a PNG with one histogram subplot per well for `variable` (not aggregated)."""
    fig, axes = plt.subplots(len(wells), 1, figsize=(8, 2.2 * len(wells)), sharex=False)
    for ax, (name, df) in zip(axes, wells.items(), strict=True):
        ax.hist(df[variable], bins=60, color="#2b6cb0")
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("frecuencia", fontsize=8)
    axes[-1].set_xlabel(variable, fontsize=9)
    fig.suptitle(f"Distribucion de {variable} por pozo (sin agregar)", fontsize=11)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"distribution_{variable}.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _find_zero_runs(is_zero: pd.Series) -> list[int]:
    """Return the length of each contiguous run of True values in a boolean series."""
    if not is_zero.any():
        return []
    change = is_zero.astype(int).diff().fillna(is_zero.iloc[0]).ne(0)
    group_id = change.cumsum()
    run_sums = is_zero.astype(int).groupby(group_id).sum()
    return [int(x) for x in run_sums[run_sums > 0].tolist()]


@dataclass(frozen=True)
class ZeroRpmSummary:
    well: str
    rows: int
    zero_rpm_rows: int
    zero_rpm_pct: float
    wob_mean_at_zero_rpm: float | None
    rop_mean_at_zero_rpm: float | None
    rop_pct_below_0_5_at_zero_rpm: float | None
    contiguous_runs: list[int]


@dataclass(frozen=True)
class ZeroGrSummary:
    well: str
    zero_gr_rows: int
    contiguous_runs: list[int]
    overlap_with_zero_rpm: int
    md_range_at_zero_gr: tuple[float, float] | None


@dataclass(frozen=True)
class LowRopSummary:
    well: str
    rop_below_1_rows: int
    rop_below_1_pct: float
    stopped_like_rows: int
    active_slow_drilling_rows: int


def summarize_zero_rpm(wells: dict[str, pd.DataFrame]) -> list[ZeroRpmSummary]:
    """For each well, characterize RPM==0 rows: are WOB/ROP also ~0 (stopped) or active?"""
    summaries = []
    for name, df in wells.items():
        mask = df["RPM"] == 0
        n_zero = int(mask.sum())
        if n_zero == 0:
            summaries.append(ZeroRpmSummary(name, len(df), 0, 0.0, None, None, None, []))
            continue
        subset = df[mask]
        summaries.append(
            ZeroRpmSummary(
                well=name,
                rows=len(df),
                zero_rpm_rows=n_zero,
                zero_rpm_pct=round(100 * n_zero / len(df), 4),
                wob_mean_at_zero_rpm=round(float(subset["WOB"].mean()), 3),
                rop_mean_at_zero_rpm=round(float(subset["ROP"].mean()), 3),
                rop_pct_below_0_5_at_zero_rpm=round(100 * float((subset["ROP"] < 0.5).mean()), 2),
                contiguous_runs=_find_zero_runs(mask),
            )
        )
    return summaries


def summarize_zero_gr(wells: dict[str, pd.DataFrame]) -> list[ZeroGrSummary]:
    """For each well, characterize GR==0 rows: contiguity, depth range, overlap with RPM==0."""
    summaries = []
    for name, df in wells.items():
        mask = df["GR"] == 0
        n_zero = int(mask.sum())
        if n_zero == 0:
            summaries.append(ZeroGrSummary(name, 0, [], 0, None))
            continue
        subset = df[mask]
        overlap = int((mask & (df["RPM"] == 0)).sum())
        summaries.append(
            ZeroGrSummary(
                well=name,
                zero_gr_rows=n_zero,
                contiguous_runs=_find_zero_runs(mask),
                overlap_with_zero_rpm=overlap,
                md_range_at_zero_gr=(
                    round(float(subset["MD"].min()), 1),
                    round(float(subset["MD"].max()), 1),
                ),
            )
        )
    return summaries


def summarize_low_rop(
    wells: dict[str, pd.DataFrame], threshold: float = 1.0
) -> list[LowRopSummary]:
    """For each well, characterize ROP<threshold rows: stopped-like vs active slow drilling."""
    summaries = []
    for name, df in wells.items():
        mask = df["ROP"] < threshold
        n_low = int(mask.sum())
        if n_low == 0:
            summaries.append(LowRopSummary(name, 0, 0.0, 0, 0))
            continue
        subset = df[mask]
        stopped_like = int(((subset["WOB"] < 0.5) & (subset["RPM"] < 1)).sum())
        active_slow = int(((subset["WOB"] >= 0.5) & (subset["RPM"] >= 1)).sum())
        summaries.append(
            LowRopSummary(
                well=name,
                rop_below_1_rows=n_low,
                rop_below_1_pct=round(100 * n_low / len(df), 4),
                stopped_like_rows=stopped_like,
                active_slow_drilling_rows=active_slow,
            )
        )
    return summaries


def main(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Generate all M2 EDA plots and the zero-value investigation JSON summary."""
    wells = load_wells(raw_data_dir)

    for variable in DISTRIBUTION_VARIABLES:
        path = plot_distribution_by_well(wells, variable, output_dir)
        logger.info("Guardado: %s", path)

    summary = {
        "rpm_zero": [asdict(s) for s in summarize_zero_rpm(wells)],
        "gr_zero": [asdict(s) for s in summarize_zero_gr(wells)],
        "rop_below_1": [asdict(s) for s in summarize_low_rop(wells)],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "zero_value_investigation.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Guardado: %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
