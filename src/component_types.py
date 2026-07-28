"""ISA-style tag prefix -> component type mapping, shared by extraction and cross-referencing.

Instrument abbreviations (PSV, FT, LSH, TIC, ...) are sourced from Kimray's
"P&ID Reference Guide" (based on ANSI/ISA-5.1-2009 Instrumentation Symbols and
Identification): https://kimray.com/sites/default/files/uploads/training-demos/
Kimray%20How%20to%20Read%20an%20Oil%20&%20Gas%20P&ID%20Reference%20Guide.pdf
— the assignment's own reference guide, and the same standard the real
Interface P&ID's tags already follow.

Equipment prefixes (P=pump, T=tank, E=exchanger, C=compressor, MV=manual/motor
valve) are a separate convention from ISA instrument tags — company/project
specific, not part of S5.1. Notably: on the real Interface P&ID, bare "V-" is
the equipment prefix for *Vessel* (V-745 = NGL Stabilizer Tower), not a valve —
confirmed against the reference guide, where "V" as an *instrument* first
letter means Vibration, not Valve either. Real valves in that document are
tagged "MV-" (motor/manual valve). "V"->valve is kept below as the common
simplified-convention default (and matches the synthetic dev fixture, built
before this reference guide was available) — MV is the one actually verified
against real data and the reference guide.
"""
import re

TAG_CORE = r"[A-Z]{1,4}-\d{2,4}[A-Z]?"  # bare, ungrouped — embed this when composing larger patterns
TAG_PATTERN = re.compile(rf"\b({TAG_CORE})\b")

TAG_PREFIX_TYPES = {
    # Equipment (company/project convention, not ISA S5.1)
    "MV": "valve",
    "PSV": "safety_valve",
    "PRV": "relief_valve",
    "V": "valve",  # default simplified convention — see module docstring caveat
    "P": "pump",
    "T": "tank",
    "C": "compressor",
    "E": "heat_exchanger",
    "AC": "heat_exchanger",  # e.g. AC-746 "after cooler" on the real document
    "F": "filter",
    # Instrument abbreviations (ANSI/ISA-5.1-2009, via Kimray reference guide)
    "FT": "flow_transmitter",
    "FI": "flow_indicator",
    "FIC": "flow_indicator_controller",
    "FIT": "flow_indicator_transmitter",
    "FC": "flow_controller",
    "FR": "flow_recorder",
    "FE": "flow_element",
    "PT": "pressure_transmitter",
    "PI": "pressure_indicator",
    "PIC": "pressure_indicator_controller",
    "PIT": "pressure_indicator_transmitter",
    "PC": "pressure_controller",
    "PCV": "pressure_control_valve",
    "PDI": "pressure_differential_indicator",
    "PDT": "pressure_differential_transmitter",
    "TT": "temperature_transmitter",
    "TI": "temperature_indicator",
    "TIC": "temperature_indicator_controller",
    "TIT": "temperature_indicator_transmitter",
    "TC": "temperature_controller",
    "TW": "thermowell",
    "LT": "level_transmitter",
    "LI": "level_indicator",
    "LIC": "level_indicator_controller",
    "LC": "level_controller",
    "LG": "level_gauge",
    "LSH": "level_switch_high",
    "LSL": "level_switch_low",
    "LAH": "level_alarm_high",
    "LAL": "level_alarm_low",
    "ESD": "emergency_shutdown_valve",
    "HOA": "hand_off_auto_switch",
    "DPI": "pressure_differential_indicator",
    "DPT": "pressure_differential_transmitter",
    "DPIT": "pressure_differential_indicator_transmitter",
}

SHAPE_TO_TYPE = {
    "circle": "instrument",
    "rectangle": "tank",
    "bowtie": "valve",
    "circle_triangle": "pump",
}


def type_from_tag(tag: str) -> str | None:
    """Infer component type from a tag like 'FT-101', 'V-12', or a space-separated
    instrument bubble tag like 'PI 715A' (see module docstring — real instrument
    tags on the reference-guide-documented bubble layout aren't always hyphenated)."""
    head = re.split(r"[\s-]", tag.strip(), maxsplit=1)[0].upper()
    prefix = head.rstrip("0123456789") or head
    return TAG_PREFIX_TYPES.get(prefix)


COARSE_CATEGORY = {
    "safety_valve": "valve",
    "flow_transmitter": "instrument",
    "pressure_transmitter": "instrument",
    "temperature_transmitter": "instrument",
    "level_transmitter": "instrument",
    "instrument": "instrument",
    "valve": "valve",
    "pump": "pump",
    "tank": "tank",
    "compressor": "compressor",
    "heat_exchanger": "heat_exchanger",
}

SOP_TYPE_KEYWORDS = {
    "valve": "valve",
    "pump": "pump",
    "tank": "tank",
    "vessel": "tank",
    "transmitter": "instrument",
    "sensor": "instrument",
    "compressor": "compressor",
    "exchanger": "heat_exchanger",
}


def coarse_category(component_type: str | None) -> str | None:
    if component_type is None:
        return None
    return COARSE_CATEGORY.get(component_type, component_type)


# This document's own ground truth (evaluation/ground_truth_page*.json) uses
# exactly 3 categories: valve/instrument/equipment. coarse_category() above
# deliberately does NOT collapse to that scheme -- crossref/compare.py needs
# the finer pump/tank/compressor/heat_exchanger split preserved to compare
# against the SOP's own declared_types vocabulary, and collapsing it there
# would silently break TYPE_MISMATCH detection. This is the separate
# 3-category collapse, used wherever code needs to score/compare against
# this project's ground truth specifically (graph_builder.py's tag-vs-
# geometry trust check, evaluation/score_cv_accuracy.py's category-coverage
# table, evaluation/score_bbox_precision.py's category matching).
#
# Real bug this closes (found live): score_cv_accuracy.py used to carry its
# own small, incomplete local mapping (4 entries) that only covered a few of
# TAG_PREFIX_TYPES' ~44 fine-grained values -- once llm_ocr_assist started
# correctly resolving real ISA tags (e.g. "PI 715A" -> "pressure_indicator"),
# those newly-legitimate fine subtypes fell through the incomplete map and
# vanished from the "instrument" row of the coverage table entirely (measured:
# page 0 instrument coverage showed 2/6 despite all 6 real instruments having
# been correctly tag-resolved -- 4 of the 6 were real "pressure_indicator"/
# "pressure_differential_indicator" resolutions, invisible to the old map).
# Pattern-matched rather than an explicit lookup table so it doesn't drift out
# of sync as TAG_PREFIX_TYPES grows.
_PROJECT_EQUIPMENT_TYPES = {"tank", "pump", "compressor", "heat_exchanger", "filter"}


def project_coarse_category(component_type: str | None) -> str | None:
    if component_type is None or component_type == "unknown":
        return None
    if "valve" in component_type:
        return "valve"
    if component_type in _PROJECT_EQUIPMENT_TYPES:
        return "equipment"
    return "instrument"
