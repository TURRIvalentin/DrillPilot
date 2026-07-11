"""M8: pure formatting of the two confidence flags M7 exposes
(known_limitation_zone, insufficient_history) into human-readable warnings. Kept
separate from app.py so this logic is testable without Streamlit's runtime, and so
the wording lives in exactly one place instead of being duplicated between the
/predict and /explain views.
"""

from __future__ import annotations

KNOWN_LIMITATION_ZONE_MESSAGE = (
    "Profundidad dentro de la zona de alto error conocida (634-988 m MD) -- "
    "confiar menos en esta prediccion. Ver docs/m6_results.md."
)

INSUFFICIENT_HISTORY_MESSAGE = (
    "Ventana con menos de 10 lecturas -- las features de ventana (medias moviles "
    "de WOB y RPM) estan en su regimen degradado de poco historial. Ver "
    "docs/adr/004-inference-input-contract.md."
)


def confidence_warnings(*, known_limitation_zone: bool, insufficient_history: bool) -> list[str]:
    """One warning message per active flag, in a fixed order, empty list if
    neither is active."""
    warnings: list[str] = []
    if known_limitation_zone:
        warnings.append(KNOWN_LIMITATION_ZONE_MESSAGE)
    if insufficient_history:
        warnings.append(INSUFFICIENT_HISTORY_MESSAGE)
    return warnings
