"""Column-name mapping between raw USROP CSV headers and short internal codes.

Shared between ml/eda and ml/cleaning so both operate on the same column names
instead of each defining its own copy. See docs/data_dictionary.md for the
full description of each code (unit, meaning, observed range).
"""

from __future__ import annotations

COLUMN_RENAME: dict[str, str] = {
    "Measured Depth m": "MD",
    "Weight on Bit kkgf": "WOB",
    "Average Standpipe Pressure kPa": "SPP",
    "Average Surface Torque kN.m": "T",
    "Rate of Penetration m/h": "ROP",
    "Average Rotary Speed rpm": "RPM",
    "Mud Flow In L/min": "FR",
    "Mud Density In g/cm3": "DS",
    "Diameter mm": "HD",
    "Average Hookload kkgf": "HL",
    "Hole Depth (TVD) m": "VD",
    "USROP Gamma gAPI": "GR",
}
