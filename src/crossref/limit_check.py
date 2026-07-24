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
# Rounding absorbs float noise at the tolerance boundary (e.g. 128.3 - 127.8
# evaluates to 0.5000000000000142, not 0.5, in raw float subtraction).
FLOAT_COMPARISON_DECIMALS = 6


def _digits_only(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    return digits or None


def _has_range_hyphen(value: str) -> bool:
    """True if value has a hyphen that isn't just a leading negative sign —
    e.g. "1-25" (a range) vs "-20" (a signed number). Deliberately narrow: a
    "/" is NOT treated as a range marker here even though this drawing set
    uses it that way sometimes ("-20/400" == "-20 to 400"), because OCR also
    inserts a stray "/" mid-digit as noise ("375" -> "3/75") — conflating the
    two would trade one false-negative for a worse false-positive on already
    real-data-validated behavior. See README for the tradeoff."""
    body = value.strip()
    rest = body[1:] if body.startswith("-") else body
    return "-" in rest


def _pressure_matches(pid_value: float | None, sop_rows: list[EquipmentLimit]) -> bool:
    if pid_value is None:
        return True  # nothing to check against
    return any(
        row.pressure_psig is not None
        and round(abs(row.pressure_psig - pid_value), FLOAT_COMPARISON_DECIMALS) <= PRESSURE_TOLERANCE_PSIG
        for row in sop_rows
    )


def _temperature_matches(pid_value: str | None, sop_rows: list[EquipmentLimit]) -> bool:
    if pid_value is None:
        return True
    pid_digits = _digits_only(pid_value)
    pid_is_range = _has_range_hyphen(pid_value)
    for row in sop_rows:
        if row.temperature_f is None:
            continue
        if _digits_only(row.temperature_f) != pid_digits:
            continue
        if _has_range_hyphen(row.temperature_f) != pid_is_range:
            continue  # same digits, different shape (e.g. "125" vs "1-25") — not a real match
        return True
    return False


def cross_reference_limits(
    pid_limits: dict[str, dict], sop_limits: dict[str, list[EquipmentLimit]]
) -> list[Discrepancy]:
    findings: list[Discrepancy] = []

    for tag in sorted(pid_limits):
        pid = pid_limits[tag]
        pid_pressure = pid.get("design_pressure_psig")
        pid_temperature = pid.get("design_temperature_f")
        sop_rows = sop_limits.get(tag)
        if not sop_rows:
            continue  # no SOP limit on record for this tag — nothing to compare

        sop_summary = ", ".join(f"{r.pressure_psig} psig / {r.temperature_f}°F" for r in sop_rows)

        if not _pressure_matches(pid_pressure, sop_rows):
            findings.append(
                Discrepancy(
                    SEVERITY_ERROR,
                    "DESIGN_LIMIT_MISMATCH",
                    f"{tag}: P&ID states design pressure {pid_pressure} psig, "
                    f"SOP table lists {sop_summary}.",
                )
            )

        if not _temperature_matches(pid_temperature, sop_rows):
            findings.append(
                Discrepancy(
                    SEVERITY_WARNING,  # temperature OCR proved noisier than pressure on this drawing
                    "DESIGN_LIMIT_MISMATCH",
                    f"{tag}: P&ID states design temperature {pid_temperature}°F, "
                    f"SOP table lists {sop_summary} (temperature OCR is lower-confidence — verify manually).",
                )
            )

    return findings
