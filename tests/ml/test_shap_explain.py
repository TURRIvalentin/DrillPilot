"""Tests for the pure/reusable helpers in ml.explainability.shap_explain. The plotting
and SHAP-computation orchestration is verified by a real run against the M4 candidate
model (see docs/m5_results.md), not unit tests -- consistent with the EDA/diagnostic
scripts elsewhere in this project (ml/eda, ml/training/diagnose_*).
"""

from __future__ import annotations

import pandas as pd

from ml.explainability.shap_explain import REGIME_GAP_MD_RANGE_M, regime_label


def test_regime_label_classifies_known_wells() -> None:
    well_id = pd.Series([0, 1, 2, 3, 4, 5, 6])

    labels = regime_label(well_id)

    assert labels.tolist() == [
        "atipico",
        "atipico",
        "dominante",
        "dominante",
        "dominante",
        "dominante",
        "atipico",
    ]


def test_regime_gap_md_range_is_the_documented_cv_pool_gap() -> None:
    low, high = REGIME_GAP_MD_RANGE_M
    assert low < high
    # Matches docs/m4_results.md: atypical CV-pool wells (1, 6) top out at 634 m,
    # dominant CV-pool wells (2, 4) start at 988 m.
    assert low == 634.0
    assert high == 988.0
