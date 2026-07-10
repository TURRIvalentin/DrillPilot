"""Dummy baseline (M4): predicts a single global mean of ROP, blind to every input.

Not segmented by regime. A regime-aware dummy (e.g., predict the dominant-regime mean
for dominant-regime rows) would require knowing a test row's well_id/regime ahead of
time -- information no genuinely deployed model has, since well_id is deliberately
excluded from the feature matrix (see docs/feature_dictionary.md). Using one blind
global mean and then reporting its MAE per regime (via ml.evaluation.metrics) is itself
informative: it should do reasonably on the dominant regime (which dominates the
training mean) and comparatively worse on the atypical regime -- exactly the gap
ADR-003 wants visible, not hidden by giving the dummy information a real model wouldn't
have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GlobalMeanBaseline(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Predicts `mean(y_train)` for every row, regardless of input."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GlobalMeanBaseline:
        self.mean_: float = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.mean_)
