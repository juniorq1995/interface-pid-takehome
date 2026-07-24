"""ISA-style tag prefix -> component type mapping, shared by extraction and cross-referencing."""
import re

TAG_CORE = r"[A-Z]{1,4}-\d{2,4}[A-Z]?"  # bare, ungrouped — embed this when composing larger patterns
TAG_PATTERN = re.compile(rf"\b({TAG_CORE})\b")

TAG_PREFIX_TYPES = {
    "PSV": "safety_valve",
    "FT": "flow_transmitter",
    "PT": "pressure_transmitter",
    "TT": "temperature_transmitter",
    "LT": "level_transmitter",
    "V": "valve",
    "P": "pump",
    "T": "tank",
    "C": "compressor",
    "E": "heat_exchanger",
}

# Longest prefix first so "PSV" matches before "P" or "V".
ORDERED_PREFIXES = sorted(TAG_PREFIX_TYPES, key=len, reverse=True)

SHAPE_TO_TYPE = {
    "circle": "instrument",
    "rectangle": "tank",
    "bowtie": "valve",
    "circle_triangle": "pump",
}


def type_from_tag(tag: str) -> str | None:
    """Infer component type from an ISA tag like 'FT-101' or 'V-12'."""
    prefix = tag.split("-")[0].upper() if "-" in tag else tag.rstrip("0123456789").upper()
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
