"""Build the combined, well-tagged, cleaned USROP dataset used by the feature pipeline.

Combines ml.cleaning.clean_usrop (per-well M2 cleaning rules) with an explicit
well_id column parsed from each raw filename, following the naming convention
documented in docs/data_dictionary.md (`USROP_A{revision} {well_id} {formation}.csv`).
Downstream code (ml.features.split, ml.features.pipeline) groups by this well_id
instead of re-deriving well identity from row order or file order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.cleaning.clean_usrop import DEFAULT_RAW_DATA_DIR, clean_well


def parse_well_id(filename: str) -> int:
    """Extract the well index from a USROP filename, e.g. 'USROP_A 3 N-SH-F-15d.csv' -> 3."""
    parts = filename.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"No se pudo extraer el well_id de '{filename}'")
    return int(parts[1])


def load_combined_dataset(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Load, clean (ml.cleaning) and well_id-tag every raw USROP well into one DataFrame.

    Row order within each well is preserved exactly as in the source CSV -- required
    for the window features in ml.features.pipeline, which depend on row order matching
    drilling sequence (increasing MD).
    """
    raw_paths = sorted(raw_data_dir.glob("*.csv"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No se encontraron CSV en {raw_data_dir}. Corre ml/data/download_usrop.py primero."
        )

    frames = []
    for path in raw_paths:
        well_id = parse_well_id(path.name)
        cleaned = clean_well(path)
        cleaned.insert(0, "well_id", well_id)
        frames.append(cleaned)
    return pd.concat(frames, ignore_index=True)
