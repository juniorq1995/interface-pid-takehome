"""Equipment discovery via label-text anchoring, not symbol-shape hunting.

Real gap this closes: pump/exchanger/cooler symbols (P-745, E-742, AC-746)
are compound, fragmented shapes with no clean geometric signature -- see
vector_equipment.py's module docstring for the F-715/V-745 signature-search
investigation, and its later note on E-742/AC-746 being tried and rejected
for the same reason. Confirmed live for P-745 specifically: zero shapes from
any existing detector (raster/vector/YOLO) fall within 95px of its real
location -- not a false negative, a symbol type nothing in this pipeline was
ever built to see geometrically.

But every equipment symbol on this document has a distinctive, reliably
OCR-able underlined text label nearby ("P-745", "E-742", "AC-746"). This
flips the discovery order: find the label first (cheap, deterministic,
whole-page Tesseract OCR), then treat its neighborhood as a candidate
equipment region, instead of trying to find the symbol's own geometry.

Deliberately coarse localization: the label's position relative to its
symbol varies by drawing convention -- below for P-745, left for F-715A,
above for AC-746 -- no single fixed offset holds across all of them. This
returns a generous region around the label rather than claiming a tight
symbol bbox. Good enough for category-coverage counting and as an anchor
for symbol_type_classifier.py's vision-model confirmation; not claimed as
localization-precision-grade output (see README "Real bbox localization
precision" for that separate, narrower metric).
"""
from __future__ import annotations

import re

import numpy as np
import pytesseract

from src.component_types import TAG_PREFIX_TYPES
from src.pid_extraction.shape_detection import DetectedShape

_EQUIPMENT_PREFIXES = {
    prefix for prefix, t in TAG_PREFIX_TYPES.items()
    if t in {"pump", "tank", "compressor", "heat_exchanger", "filter"}
}
# Tolerant separator, not a literal hyphen: confirmed live that Tesseract
# reads this document's tag hyphens as "=" often enough to matter (P-745 ->
# "P=745" -- the exact reason a naive TAG_PATTERN search missed the real
# pump label entirely on first attempt). Same tolerance title_block.py's
# PREFIX_PATTERN already uses for the same documented reason. This is a
# location-discovery pattern only -- looser than the real tag-accuracy
# patterns on purpose, since the goal here is finding *where* to look, not
# the exact tag text (that still goes through the stricter OCR/LLM tag
# reading paths once a shape exists to attach a tag to).
_TOLERANT_TAG_PATTERN = re.compile(r"\b([A-Z]{1,4})[-=/](\d{2,4})\b")
_REGION_HALF_WIDTH_PX = 220
_REGION_HALF_HEIGHT_PX = 220
# Checked as bbox-containment-with-margin against already-known *equipment*
# shapes (e.g. vector_equipment.py's vessel matches), not a simple
# center-distance circle -- a tall vessel's own label sits near one end
# (e.g. F-715A/B's label is near the top of a 437px-tall body), far outside
# any reasonable radius from the shape's geometric center. Not checked
# against valves/instruments at all -- those legitimately cluster right next
# to equipment and must not suppress a real label discovery.
_EXISTING_EQUIPMENT_MARGIN_PX = 150
_DEDUP_RADIUS_PX = 60


def _near_existing_equipment(cx: float, cy: float, existing_equipment_shapes: list[DetectedShape]) -> bool:
    for shape in existing_equipment_shapes:
        x, y, w, h = shape.bbox
        if (x - _EXISTING_EQUIPMENT_MARGIN_PX) <= cx <= (x + w + _EXISTING_EQUIPMENT_MARGIN_PX) and \
           (y - _EXISTING_EQUIPMENT_MARGIN_PX) <= cy <= (y + h + _EXISTING_EQUIPMENT_MARGIN_PX):
            return True
    return False


def find_equipment_label_candidates(
    image: np.ndarray,
    existing_equipment_shapes: list[DetectedShape],
    start_shape_id: int = 0,
) -> list[DetectedShape]:
    """Scan the page for equipment-prefix tag text not already covered by an
    already-detected equipment shape, return a coarse candidate region
    around each real hit found."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    ih, iw = image.shape[:2]

    candidates: list[DetectedShape] = []
    seen_positions: list[tuple[float, float]] = []
    next_id = start_shape_id

    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        match = _TOLERANT_TAG_PATTERN.search(word.upper())
        if not match:
            continue
        prefix = match.group(1)
        if prefix not in _EQUIPMENT_PREFIXES:
            continue

        wx, wy, ww, wh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        cx, cy = wx + ww / 2.0, wy + wh / 2.0

        if _near_existing_equipment(cx, cy, existing_equipment_shapes):
            continue
        if any(((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 < _DEDUP_RADIUS_PX for sx, sy in seen_positions):
            continue
        seen_positions.append((cx, cy))

        x0 = max(0, int(cx - _REGION_HALF_WIDTH_PX))
        y0 = max(0, int(cy - _REGION_HALF_HEIGHT_PX))
        x1 = min(iw, int(cx + _REGION_HALF_WIDTH_PX))
        y1 = min(ih, int(cy + _REGION_HALF_HEIGHT_PX))
        if x1 <= x0 or y1 <= y0:
            continue

        candidates.append(
            DetectedShape(
                shape_id=next_id,
                kind="equipment_label_region",
                bbox=(x0, y0, x1 - x0, y1 - y0),
                center=(int(cx), int(cy)),
                contour=np.empty((0, 1, 2), dtype=np.int32),
            )
        )
        next_id += 1

    return candidates
