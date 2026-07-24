"""Parse equipment design/operating limits out of an SOP table.

The real Interface SOP document turned out to be a design-limits table
(equipment name -> pressure, temperature), not a narrative procedure — see
README "Tested Against Real Data". This is a separate extraction path from
tag_extractor.py's narrative regex parsing; both run and their results merge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TAG_AT_START = re.compile(r"^([A-Z]{1,4}-\d{2,4})")


@dataclass
class EquipmentLimit:
    tag: str
    raw_label: str
    pressure_psig: float | None
    temperature_f: str | None


def _to_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def extract_equipment_limits(tables: list[list[list[str]]]) -> dict[str, list[EquipmentLimit]]:
    """Look for a table with a label column plus pressure/temperature columns.

    Returns a tag -> list of limits, since one tag can have multiple rows (e.g. a
    heat exchanger's shell-side and tube-side limits differ but share a tag)."""
    limits: dict[str, list[EquipmentLimit]] = {}

    for table in tables:
        if len(table) < 2:
            continue
        header = [c.lower() for c in table[0]]
        pressure_col = next((i for i, c in enumerate(header) if "pressure" in c), None)
        temp_col = next((i for i, c in enumerate(header) if "temp" in c), None)
        if pressure_col is None and temp_col is None:
            continue

        for row in table[1:]:
            label = row[0].strip() if row else ""
            match = TAG_AT_START.match(label.upper())
            if not match:
                continue
            tag = match.group(1)
            pressure = _to_float(row[pressure_col]) if pressure_col is not None and pressure_col < len(row) else None
            temperature = row[temp_col].strip() if temp_col is not None and temp_col < len(row) else None
            limits.setdefault(tag, []).append(
                EquipmentLimit(tag=tag, raw_label=label, pressure_psig=pressure, temperature_f=temperature or None)
            )

    return limits
