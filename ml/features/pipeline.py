"""Sklearn-compatible feature engineering pipeline for USROP (M3).

Turns the cleaned, well-tagged USROP dataset (ml.features.dataset.load_combined_dataset)
into a model-ready feature matrix. Designed to be serialized together with the trained
model as one artifact -- no parallel feature logic re-implemented in the backend later
(fixed in the project's original architecture, see the top-level design).

Direct features: the 11 USROP predictor columns (MD, WOB, SPP, T, RPM, FR, DS, HD, HL,
VD, GR) plus gr_imputed (added by ml.cleaning). ROP is the target (y), well_id is a
bookkeeping/grouping column -- neither is ever a model feature. Using well_id as a
feature would leak well identity and be meaningless for a genuinely new well at
inference time.

Window features are a small, physically justified set -- see docs/feature_dictionary.md
for the reasoning behind each one. All of them are backward-looking only (no centered
windows, no negative shifts) and grouped by well_id before any rolling/diff, per
docs/adr/003-split-strategy.md decision #3. tests/ml/test_features.py enforces both
properties directly.

This transformer learns nothing from the data it is fit on (no per-well or per-split
statistics are stored) -- every feature is computed independently per row from that
row's own well history. There is therefore no leakage risk in reusing the same fitted
transformer across the CV pool and the held-out test set.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DIRECT_FEATURE_COLUMNS: tuple[str, ...] = (
    "MD",
    "WOB",
    "SPP",
    "T",
    "RPM",
    "FR",
    "DS",
    "HD",
    "HL",
    "VD",
    "GR",
    "gr_imputed",
)

_REQUIRED_INPUT_COLUMNS = ("well_id", *DIRECT_FEATURE_COLUMNS)


class USROPFeatureTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Direct + backward-looking window features for USROP, grouped by well_id.

    Parameters
    ----------
    rolling_window:
        Number of preceding rows (within the same well) used for the rolling-window
        features. Default 10.
    """

    def __init__(self, rolling_window: int = 10) -> None:
        self.rolling_window = rolling_window

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> USROPFeatureTransformer:
        """Validate `X`. Stateless: no statistics are learned or stored."""
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the feature matrix for `X`, in the same row order as `X`."""
        self._validate_input(X)
        grouped = X.groupby("well_id", sort=False)
        window = self.rolling_window

        features = X[list(DIRECT_FEATURE_COLUMNS)].copy()
        features["gr_imputed"] = features["gr_imputed"].astype(int)

        features[f"WOB_rolling_mean_{window}"] = grouped["WOB"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        features[f"RPM_rolling_mean_{window}"] = grouped["RPM"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        features[f"T_rolling_std_{window}"] = (
            grouped["T"]
            .transform(lambda s: s.rolling(window=window, min_periods=1).std())
            .fillna(0.0)
        )
        features["WOB_diff_1"] = grouped["WOB"].transform(lambda s: s.diff(1)).fillna(0.0)

        return features.reset_index(drop=True)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        window = self.rolling_window
        return [
            *DIRECT_FEATURE_COLUMNS,
            f"WOB_rolling_mean_{window}",
            f"RPM_rolling_mean_{window}",
            f"T_rolling_std_{window}",
            "WOB_diff_1",
        ]

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en X: {missing}")

        md_diff = X.groupby("well_id", sort=False)["MD"].diff()
        if (md_diff.dropna() < 0).any():
            raise ValueError(
                "MD no es no-decreciente dentro de al menos un well_id -- las features "
                "de ventana requieren que las filas estén ordenadas dentro de cada pozo."
            )
