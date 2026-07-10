"""Leave-one-well-out cross-validation splitting, per docs/adr/003-split-strategy.md.

Shared by ml.models.byoung_reduced (grid search over the WOB threshold constant) and
ml.models.lightgbm_model (Optuna hyperparameter tuning) so both use the exact same CV
folds -- never GroupKFold, never a random split.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def leave_one_well_out_splits(
    well_id: pd.Series,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield (train_positions, val_positions, held_out_well_id) for each well in `well_id`.

    One fold per distinct well present in `well_id`: that well's rows become the
    validation set, every other row becomes train. Positions are integer (0-based,
    positional) indices into `well_id`, suitable for `.iloc[]` -- not label-based index
    values -- so callers don't need `well_id` to have a default RangeIndex.
    """
    wells = sorted(well_id.unique())
    positions = np.arange(len(well_id))
    well_id_array = well_id.to_numpy()

    for held_out in wells:
        val_mask = well_id_array == held_out
        yield positions[~val_mask], positions[val_mask], int(held_out)
