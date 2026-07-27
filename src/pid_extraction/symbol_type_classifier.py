"""Phase 1: LLM-assisted symbol *type* classification, for shapes the
geometric heuristics in shape_detection.py couldn't classify at all.

Same recipe and same local model as llm_ocr_assist.py (crop -> local Ollama
vision model), applied to a different question: not "what does the tag say"
but "what kind of P&ID symbol is this." Targets exactly the shapes that reach
graph_builder.py with component_type == "unknown" -- SHAPE_TO_TYPE only
covers 4 raster kinds (circle/rectangle/bowtie/circle_triangle); anything
shape_detection.py couldn't fit one of those falls through to "unknown" with
no further attempt to say what it actually is. This was flagged as "not yet
built" in README's "Next: Phase 1" section — this closes that gap.

Deliberately narrow scope: classifies into this project's existing coarse
categories (valve/instrument/equipment) plus a best-effort ISA-style subtype
label when the model is confident enough to name one (e.g. "gate_valve",
"pressure_indicator") -- it does not invent a new taxonomy. A subtype the
model returns that isn't in _KNOWN_SUBTYPES is kept as coarse-category-only
rather than trusted verbatim, since an unrecognized string from the model is
more likely a hallucinated/nonstandard label than a real gap in this list.

Local/open-source by design (Ollama, not a hosted API) — no data leaves the
machine, no API key, no per-call cost. Opt-in and additive: only invoked for
shapes still "unknown" after tag-based and YOLO-based typing, so it never
overrides a type resolved by a cheaper/more-certain path.
"""
from __future__ import annotations

import re

import numpy as np
import requests

from src.component_types import TAG_PREFIX_TYPES
from src.pid_extraction.llm_ocr_assist import OLLAMA_URL, REQUEST_TIMEOUT_S, VISION_MODEL, _encode_crop
from src.pid_extraction.ocr_tagging import _crop_label_region
from src.pid_extraction.shape_detection import DetectedShape

# Reuse the same fine-grained vocabulary tag prefixes already resolve to, so a
# vision-classified subtype and a tag-resolved subtype mean the same thing
# downstream (e.g. both produce "pressure_indicator", never two spellings of
# the same concept) -- plus common valve-body subtypes real tag prefixes on
# this document structurally can't distinguish (MV covers every manual valve
# regardless of body type), which is exactly the gap visual classification is
# suited to close. Source for the valve-subtype names: the Roboflow "P&ID
# Symbols R2" dataset already staged under datasets/ (see README "Datasets
# pulled for future symbol-detection work"), the natural reference vocabulary
# for this since it has real class names rather than requiring one be invented.
_KNOWN_SUBTYPES = set(TAG_PREFIX_TYPES.values()) | {
    "valve", "instrument", "tank", "pump", "heat_exchanger",
    "gate_valve", "ball_valve", "check_valve", "globe_valve", "butterfly_valve", "needle_valve",
}
# The prompt asks for exactly one of these 4 words as the coarse category --
# note this is "equipment", not component_types.py's finer tank/pump/
# heat_exchanger/compressor split (COARSE_CATEGORY doesn't even collapse
# those to "equipment"; that collapse is score_cv_accuracy.py's own local
# scoring convention) -- kept separate here deliberately, matched to what
# this project's ground truth files actually use as their 3 categories.
_COARSE_CATEGORIES = {"valve", "instrument", "equipment"}

PROMPT = (
    "This is a cropped symbol from an industrial P&ID (piping and instrumentation "
    "diagram). Classify it into exactly one of: valve, instrument, equipment, unknown. "
    "If you can confidently name a more specific type (e.g. gate_valve, check_valve, "
    "pressure_indicator, level_indicator, pump, tank, heat_exchanger), add it after a "
    "comma. Reply with ONLY the classification, nothing else. Examples of valid replies: "
    "'valve, gate_valve' or 'instrument' or 'unknown'."
)


def _parse_classification(raw_text: str) -> tuple[str | None, str | None]:
    """Return (coarse_category_or_None, subtype_or_None) from the model's raw reply."""
    parts = [p.strip().lower() for p in raw_text.split(",")]
    if not parts or not parts[0]:
        return None, None
    coarse = parts[0]
    if coarse not in _COARSE_CATEGORIES:
        return None, None
    subtype = None
    if len(parts) > 1:
        candidate = re.sub(r"[^a-z_]", "", parts[1].replace(" ", "_"))
        if candidate in _KNOWN_SUBTYPES:
            subtype = candidate
    return coarse, subtype


def classify_symbol_type_with_llm(image: np.ndarray, shape: DetectedShape) -> tuple[str | None, str | None, str]:
    """Return (coarse_category_or_None, subtype_or_None, raw_model_text).

    Best-effort: any network/model failure or an unparseable/off-vocabulary
    reply degrades to (None, None, raw_text) rather than raising -- this is
    an optional enhancement layered over the deterministic shape/tag typing
    path, same failure contract as llm_ocr_assist.read_tag_with_llm.
    """
    crop = _crop_label_region(image, shape, margin=20)
    if crop.size == 0:
        return None, None, ""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": PROMPT,
                "images": [_encode_crop(crop)],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
    except (requests.exceptions.RequestException, ValueError):
        return None, None, ""

    coarse, subtype = _parse_classification(raw_text)
    return coarse, subtype, raw_text
