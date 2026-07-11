"""Bourgoyne & Young reducido (4/8 terminos) -- M4 physical baseline.

Only 3 of the original 8 Bourgoyne & Young (1974) terms (plus the intercept a1) are
computable from USROP -- see docs/adr/002-modelo-baseline.md for the full 8-term
mapping and why x3, x4, x7 and x8 are omitted. Always refer to this model as
"Bourgoyne & Young reducido (4/8 terminos)" in logs, filenames, docstrings and plots --
never "Bourgoyne & Young" unqualified, per that ADR's explicit instruction: this is not
the industry-standard equation, it is the best approximation USROP's columns allow.

    ln(ROP) = a1 + a2*x2 + a5*x5 + a6*x6

    x2 = BY_REFERENCE_DEPTH_FT - D_ft                                  (depth / compaction)
    x5 = ln[(w_over_db - wob_threshold) / (BY_WOB_NORM_CONSTANT - wob_threshold)]  (WOB/bit-diameter)
    x6 = ln(N / SECONDS_PER_MINUTE)                                    (rotary speed)

a1, a2, a5, a6 are fit by ordinary least squares on ln(ROP) over the CV-pool (wells
1, 2, 4, 6) only -- never on test (wells 0, 3, 5). `wob_threshold` (the "(W/db)_t" term
in the literature) has no single defensible universal value (see ADR-002's note on
x5), so it is selected by grid search using leave-one-well-out CV over the same
CV-pool and minimizing MAE, instead of being hardcoded from a citation with false
precision.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from ml.training.cv import leave_one_well_out_splits

logger = logging.getLogger(__name__)

MODEL_NAME = "Bourgoyne & Young reducido (4/8 terminos)"

# Unit conversions (USROP is metric; the classic B&Y formulation is US customary).
FT_PER_METER = 3.28084
LBF_PER_KGF = 2.20462
IN_PER_MM = 1.0 / 25.4
SECONDS_PER_MINUTE = 60.0

# Literature constants (Bourgoyne & Young 1974). See docs/adr/002-modelo-baseline.md
# for the verification caveat: the model *structure* was confirmed against multiple
# sources, but these exact figures were not re-verified against the primary source in
# that session -- check before trusting for anything beyond this project's baseline.
BY_REFERENCE_DEPTH_FT = 10_000.0
BY_WOB_NORM_CONSTANT = 4.0  # units: 1000 lbf/in

# wob_threshold candidates searched via LOWO-CV (units: 1000 lbf/in). Calibrated
# against USROP's real W/db distribution (median ~1.0, p95 ~3.5, in these units) so the
# grid stays in a range where clipping (see _X5_EPSILON below) does not dominate the
# term for most candidates. No literature value is used for this constant -- see
# ADR-002.
WOB_THRESHOLD_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)

# x5's log argument is undefined (or negative) when a row's W/db falls below
# wob_threshold. Clipped to this small positive floor instead of raising -- purely a
# numerical safeguard, not a physically meaningful value. `clipped_fraction_` (set on
# fit) reports how often this triggers, so the safeguard doesn't silently hide how much
# of the term is actually degenerate for the chosen threshold.
_X5_EPSILON = 1e-6

# x6 = ln(N/60) is undefined at N=0. USROP has real RPM==0 rows -- sliding drilling
# with a downhole motor (see docs/eda_findings.md, "RPM == 0 y GR == 0"), deliberately
# NOT removed in M2 because they are legitimate active-drilling events, not sensor
# faults. The classic Bourgoyne & Young formula predates widespread mud-motor sliding
# as a routine directional-drilling technique and has no term for it -- this is a real
# domain limitation of applying the 1974 rotary-drilling model to USROP, not just a
# numerical edge case. Clipped to this floor (equivalent to an effectively-stalled
# rotary reading) so the model still produces a finite prediction for those rows;
# `rpm_clipped_fraction_` (set on fit) reports how often this triggers.
_RPM_EPSILON = 1e-3

_REQUIRED_COLUMNS = ("well_id", "MD", "WOB", "HD", "RPM")


def _compute_terms(X: pd.DataFrame, wob_threshold: float) -> pd.DataFrame:
    """Compute x2, x5, x6 from raw USROP columns, plus a clipping diagnostic for x5."""
    d_ft = X["MD"] * FT_PER_METER
    x2 = BY_REFERENCE_DEPTH_FT - d_ft

    w_lbf = X["WOB"] * 1000.0 * LBF_PER_KGF
    db_in = X["HD"] * IN_PER_MM
    w_over_db = (w_lbf / db_in) / 1000.0  # 1000 lbf/in
    raw_numerator = w_over_db - wob_threshold
    numerator = raw_numerator.clip(lower=_X5_EPSILON)
    denominator = BY_WOB_NORM_CONSTANT - wob_threshold
    x5 = np.log(numerator / denominator)

    rpm_clipped = X["RPM"] < _RPM_EPSILON
    rpm_safe = X["RPM"].clip(lower=_RPM_EPSILON)
    x6 = np.log(rpm_safe / SECONDS_PER_MINUTE)

    return pd.DataFrame(
        {
            "x2": x2.to_numpy(),
            "x5": x5.to_numpy(),
            "x6": x6.to_numpy(),
            "_clipped": (raw_numerator < _X5_EPSILON).to_numpy(),
            "_rpm_clipped": rpm_clipped.to_numpy(),
        }
    )


class BourgoyneYoungReduced(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Reduced Bourgoyne & Young (a1 + a2*x2 + a5*x5 + a6*x6), fit on USROP's CV-pool.

    X must contain well_id, MD, WOB, HD, RPM (raw cleaned USROP columns -- not the M3
    feature matrix). well_id is required here for the internal LOWO-CV threshold
    search; none of M3's window features map to a B&Y term (see ADR-002), so this
    model does not use ml.features.pipeline at all.
    """

    def __init__(self, wob_threshold_grid: tuple[float, ...] = WOB_THRESHOLD_GRID) -> None:
        self.wob_threshold_grid = wob_threshold_grid

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BourgoyneYoungReduced:
        self._validate_input(X)
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)
        log_y = np.log(y)

        best_threshold = self.wob_threshold_grid[0]
        best_cv_mae = np.inf
        search_log: list[dict[str, float]] = []

        for threshold in self.wob_threshold_grid:
            fold_maes = []
            for train_pos, val_pos, _held_out_well in leave_one_well_out_splits(X["well_id"]):
                terms_train = _compute_terms(X.iloc[train_pos], threshold)
                terms_val = _compute_terms(X.iloc[val_pos], threshold)
                reg = LinearRegression()
                reg.fit(terms_train[["x2", "x5", "x6"]], log_y.iloc[train_pos])
                pred_val = np.exp(reg.predict(terms_val[["x2", "x5", "x6"]]))
                fold_maes.append(mean_absolute_error(y.iloc[val_pos], pred_val))
            mean_cv_mae = float(np.mean(fold_maes))
            search_log.append({"wob_threshold": threshold, "cv_mae": mean_cv_mae})
            logger.info("wob_threshold=%.3f -> LOWO-CV MAE=%.4f", threshold, mean_cv_mae)
            if mean_cv_mae < best_cv_mae:
                best_cv_mae = mean_cv_mae
                best_threshold = threshold

        self.wob_threshold_: float = best_threshold
        self.threshold_search_log_: list[dict[str, float]] = search_log
        self.cv_mae_: float = best_cv_mae

        terms_full = _compute_terms(X, self.wob_threshold_)
        self.clipped_fraction_: float = float(terms_full["_clipped"].mean())
        self.rpm_clipped_fraction_: float = float(terms_full["_rpm_clipped"].mean())
        final_reg = LinearRegression()
        final_reg.fit(terms_full[["x2", "x5", "x6"]], log_y)
        self.a1_ = float(final_reg.intercept_)
        self.a2_, self.a5_, self.a6_ = (float(c) for c in final_reg.coef_)
        self._regressor = final_reg
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_input(X)
        terms = _compute_terms(X, self.wob_threshold_)
        log_rop = self._regressor.predict(terms[["x2", "x5", "x6"]])
        return np.asarray(np.exp(log_rop))

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en X: {missing}")
