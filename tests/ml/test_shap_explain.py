"""Tests for the pure/reusable helpers in ml.explainability.shap_explain. The plotting
and SHAP-computation orchestration is verified by a real run against the M4 candidate
model (see docs/m5_results.md), not unit tests -- consistent with the EDA/diagnostic
scripts elsewhere in this project (ml/eda, ml/training/diagnose_*).
"""

from __future__ import annotations

import pandas as pd

from ml.explainability.shap_explain import regime_label


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


# REGIME_GAP_MD_RANGE_M (re-exported from ml.evaluation.metrics.KNOWN_LIMITATION_MD_RANGE_M)
# is tested at its source in tests/ml/test_metrics.py, not duplicated here.
