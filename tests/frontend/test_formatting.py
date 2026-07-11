"""Tests for frontend.streamlit_app.formatting -- pure functions, no Streamlit
runtime needed."""

from __future__ import annotations

from frontend.streamlit_app.formatting import (
    INSUFFICIENT_HISTORY_MESSAGE,
    KNOWN_LIMITATION_ZONE_MESSAGE,
    confidence_warnings,
)


def test_no_warnings_when_both_flags_are_false() -> None:
    assert confidence_warnings(known_limitation_zone=False, insufficient_history=False) == []


def test_known_limitation_zone_warning_only() -> None:
    warnings = confidence_warnings(known_limitation_zone=True, insufficient_history=False)
    assert warnings == [KNOWN_LIMITATION_ZONE_MESSAGE]


def test_insufficient_history_warning_only() -> None:
    warnings = confidence_warnings(known_limitation_zone=False, insufficient_history=True)
    assert warnings == [INSUFFICIENT_HISTORY_MESSAGE]


def test_both_warnings_when_both_flags_are_true() -> None:
    warnings = confidence_warnings(known_limitation_zone=True, insufficient_history=True)
    assert warnings == [KNOWN_LIMITATION_ZONE_MESSAGE, INSUFFICIENT_HISTORY_MESSAGE]
