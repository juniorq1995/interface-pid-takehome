"""Cross-check P&ID title-block design limits against the SOP's limits table.

Separate from compare.py's graph-vs-SOP cross-referencing: this compares two
sets of *extracted facts* (P&ID header OCR vs SOP table), not the component
graph. See title_block.py and limits_table.py for how each side is produced.
"""
from __future__ import annotations

import re

from src.crossref.compare import Discrepancy, SEVERITY_ERROR, SEVERITY_WARNING
from src.sop_extraction.limits_table import EquipmentLimit

PRESSURE_TOLERANCE_PSIG = 0.5


def _digits_only(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    return digits or None


def _pressure_matches(pid_value: float | None, sop_rows: list[EquipmentLimit]) -> bool:
    if pid_value is None:
        return True  # nothing to check against
    return any(row.pressure_psig is not None and abs(row.pressure_psig - pid_value) <= PRESSURE_TOLERANCE_PSIG for row in sop_rows)


def _temperature_matches(pid_value: str | None, sop_rows: list[EquipmentLimit]) -> bool:
    if pid_value is None:
        return True
    pid_digits = _digits_only(pid_value)
    return any(_digits_only(row.temperature_f) == pid_digits for row in sop_rows)


def cross_reference_limits(
    pid_limits: dict[str, dict], sop_limits: dict[str, list[EquipmentLimit]]
) -> list[Discrepancy]:
    findings: list[Discrepancy] = []

    for tag in sorted(pid_limits):
        pid = pid_limits[tag]
        sop_rows = sop_limits.get(tag)
        if not sop_rows:
            continue  # no SOP limit on record for this tag — nothing to compare

        sop_summary = ", ".join(f"{r.pressure_psig} psig / {r.temperature_f}°F" for r in sop_rows)

        if not _pressure_matches(pid["design_pressure_psig"], sop_rows):
            findings.append(
                Discrepancy(
                    SEVERITY_ERROR,
                    "DESIGN_LIMIT_MISMATCH",
                    f"{tag}: P&ID states design pressure {pid['design_pressure_psig']} psig, "
                    f"SOP table lists {sop_summary}.",
                )
            )

        if not _temperature_matches(pid["design_temperature_f"], sop_rows):
            findings.append(
                Discrepancy(
                    SEVERITY_WARNING,  # temperature OCR proved noisier than pressure on this drawing
                    "DESIGN_LIMIT_MISMATCH",
                    f"{tag}: P&ID states design temperature {pid['design_temperature_f']}°F, "
                    f"SOP table lists {sop_summary} (temperature OCR is lower-confidence — verify manually).",
                )
            )

    return findings
