"""Apply the M2 data-cleaning rules approved in docs/cleaning_rules.md to raw USROP wells.

Rule 1: rows with RPM == 0 (sliding drilling with a downhole motor) are left unchanged.
Rule 2: rows with ROP < 1 m/h (slow active drilling) are left unchanged.
Rule 3: GR == 0 (gamma-ray telemetry dropouts) is imputed via linear interpolation within
each well, ordered by row index; edge gaps fall back to the nearest valid value. A new
`gr_imputed` boolean column marks which rows were touched, so the imputation stays
traceable -- unlike the original forward/backward filling done by Tunkiel et al. (2021),
which cannot be distinguished from measured values (see docs/data_dictionary.md).

No rows are ever dropped by this module. See docs/cleaning_rules.md for the evidence and
reasoning behind each rule, including the one long/anomalous GR gap in well 4.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.data.schema import COLUMN_RENAME

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "interim"


def load_raw_well(path: Path) -> pd.DataFrame:
    """Load a single raw USROP CSV, renaming columns to their short codes."""
    return pd.read_csv(path, index_col=0).rename(columns=COLUMN_RENAME)


def impute_gr_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 3: impute GR == 0 via linear interpolation within `df`, adding `gr_imputed`.

    Edge gaps (no valid neighbour on one side) fall back to the nearest valid value
    (equivalent to forward/backward-fill at the boundary). Never drops a row.
    """
    df = df.copy()
    df["gr_imputed"] = df["GR"] == 0
    if df["gr_imputed"].any():
        gr_with_gaps = df["GR"].mask(df["gr_imputed"])
        df["GR"] = gr_with_gaps.interpolate(method="linear", limit_direction="both")
    return df


def apply_cleaning_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 3 approved M2 cleaning rules to a single well's DataFrame.

    Rules 1 and 2 (RPM == 0, ROP < 1) are no-ops by construction: this function never
    filters or transforms rows based on RPM or ROP values. Only rule 3 (GR == 0) changes
    data.
    """
    return impute_gr_zero(df)


def clean_well(path: Path) -> pd.DataFrame:
    """Load and clean a single raw USROP well CSV."""
    return apply_cleaning_rules(load_raw_well(path))


def clean_all_wells(
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[Path]:
    """Clean every raw USROP CSV in `raw_data_dir` and write the result to `output_dir`.

    Returns the paths written. Not committed to git (data/interim is gitignored, same as
    data/raw) -- reproducible by re-running this function.
    """
    raw_paths = sorted(raw_data_dir.glob("*.csv"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No se encontraron CSV en {raw_data_dir}. Corre ml/data/download_usrop.py primero."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in raw_paths:
        cleaned = clean_well(path)
        output_path = output_dir / path.name
        cleaned.to_csv(output_path)
        logger.info(
            "Guardado: %s (%d filas, %d con GR imputado)",
            output_path,
            len(cleaned),
            int(cleaned["gr_imputed"].sum()),
        )
        written.append(output_path)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_all_wells()
