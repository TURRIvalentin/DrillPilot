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

# MD band with zero CV-pool coverage from either regime: atypical CV-pool wells
# (1, 6) top out at 634 m, dominant CV-pool wells (2, 4) start at 988 m (see
# docs/adr/003-split-strategy.md's regime experiment). Confirmed three times
# independently as a genuine high-error zone, not just a training-coverage gap:
# M4's regime router (100% CV-pool accuracy, 16.2% real accuracy on well 0), M5's
# SHAP MD dependence plot (qualitative break at this exact band), and M6's live
# inference validation (well 0 prediction MAE spikes to ~31 past this band, worse
# than the well's already-weak start). Single source of truth for this constant --
# ml.explainability.shap_explain and the backend's known_limitation_zone response
# field both import it from here. See docs/m6_results.md.
KNOWN_LIMITATION_MD_RANGE_M: tuple[float, float] = (634.0, 988.0)


def regime_of(well_id: int) -> str:
    """Classify a well_id into the two regimes established in ADR-003's context table."""
    if well_id in DOMINANT_REGIME_WELL_IDS:
        return "dominante"
    if well_id in ATYPICAL_REGIME_WELL_IDS:
        return "atipico"
    raise ValueError(f"well_id desconocido para USROP: {well_id}")


def is_in_known_limitation_zone(md: float) -> bool:
    """Whether `md` (meters) falls inside KNOWN_LIMITATION_MD_RANGE_M -- the
    documented high-error depth band. Inclusive on both ends."""
    low, high = KNOWN_LIMITATION_MD_RANGE_M
    return low <= md <= high


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
