"""MAE reporting at the three levels required by docs/adr/003-split-strategy.md:
pooled, per-well, and per-regime (dominante vs. atípico). Used to evaluate every
baseline in M4 the same way, so the comparison in docs/m4_results.md is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

DOMINANT_REGIME_WELL_IDS: frozenset[int] = frozenset({2, 3, 4, 5})
ATYPICAL_REGIME_WELL_IDS: frozenset[int] = frozenset({0, 1, 6})


def regime_of(well_id: int) -> str:
    """Classify a well_id into the two regimes established in ADR-003's context table."""
    if well_id in DOMINANT_REGIME_WELL_IDS:
        return "dominante"
    if well_id in ATYPICAL_REGIME_WELL_IDS:
        return "atipico"
    raise ValueError(f"well_id desconocido para USROP: {well_id}")


@dataclass(frozen=True)
class MaeReport:
    """MAE at the three levels ADR-003 requires. `by_well` and `by_regime` are stable-sorted."""

    pooled: float
    by_well: dict[int, float]
    by_regime: dict[str, float]

    def to_flat_metrics(self, prefix: str) -> dict[str, float]:
        """Flatten into {metric_name: value} for MLflow (e.g. 'mae_pooled', 'mae_well_3')."""
        flat = {f"{prefix}_pooled": self.pooled}
        flat.update({f"{prefix}_well_{well}": mae for well, mae in self.by_well.items()})
        flat.update({f"{prefix}_regime_{regime}": mae for regime, mae in self.by_regime.items()})
        return flat


def mae_report(y_true: pd.Series, y_pred: np.ndarray, well_id: pd.Series) -> MaeReport:
    """Compute pooled + per-well + per-regime MAE for one set of predictions."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    well_id_arr = np.asarray(well_id)

    pooled = float(mean_absolute_error(y_true_arr, y_pred_arr))

    by_well: dict[int, float] = {}
    for well in sorted(np.unique(well_id_arr)):
        mask = well_id_arr == well
        by_well[int(well)] = float(mean_absolute_error(y_true_arr[mask], y_pred_arr[mask]))

    by_regime: dict[str, float] = {}
    for regime in ("dominante", "atipico"):
        mask = np.array([regime_of(int(w)) == regime for w in well_id_arr])
        if mask.any():
            by_regime[regime] = float(mean_absolute_error(y_true_arr[mask], y_pred_arr[mask]))

    return MaeReport(pooled=pooled, by_well=by_well, by_regime=by_regime)
