"""OCR the per-sheet title/design-note block and extract stated design limits.

Discovered from testing against the real Interface P&ID (see README "Tested
Against Real Data"): each sheet has a header note block (top ~16% of the page)
naming the major equipment on that sheet and its design pressure/temperature —
e.g. "F-715 A & B ... DESIGN: 275 PSIG @ 100F". The real SOP turned out to be a
table of the same design limits per tag, so this is the real, meaningful
cross-check for this specific document pair (see limits_table.py).

Two OCR quirks found on this specific drawing, both handled below:

1. The tag line sits directly above an underline rule, and digits there (e.g.
   "715") get reliably mangled ("F-/I39") even at high DPI and after erasing
   the underline stroke — the letters do not. Since tag prefixes are unique
   within a document, matching on prefix alone against the known tag
   vocabulary supplied by the caller (normally the SOP's equipment list) is
   more robust than reconstructing exact digits from noisy OCR.
2. Sheets can name two pieces of equipment side by side, each with its own
   "MWP ... PSIG ... F" clause (see page 2: AC-746 and E-742). A single
   text-wide search would misattribute one clause's numbers to both tags, so
   tags and limit clauses are paired by left-to-right position instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from src.pid_extraction.pdf_to_image import pdf_to_images

HEADER_HEIGHT_FRACTION = 0.16
# The row-number ladder runs the full height of the page in narrow strips hugging
# the left/right border — excluding those margins keeps it out of the header crop.
HEADER_MARGIN_FRACTION = 0.04

# "—"/"–" (em/en dash) included alongside "=" and "/": Tesseract reads this
# document's tag hyphens as any of these on different labels (confirmed live
# for AC-746: "AC-746" -> "AC—746", em-dash, not a regular hyphen).
PREFIX_PATTERN = re.compile(r"\b([A-Z]{1,4})[-=/—–]\d")  # {1,4} to match TAG_CORE's prefix width (component_types.py)
PRESSURE_PATTERN = re.compile(
    r"(?:DESIGN|M\.?W\.?P\.?)[:\s]*[^0-9]{0,15}?(\d{2,4}(?:\.\d+)?)\s*PSIG", re.IGNORECASE
)
TEMPERATURE_PATTERN = re.compile(r"(-?\d{1,3}(?:\s*(?:TO|/)\s*-?\d{1,3})?)\s*.?\s*F\b", re.IGNORECASE)


def _prefix_positions(text: str, prefixes: set[str]) -> list[tuple[int, str]]:
    """Loose match: prefix + a dash-like separator (tolerant of OCR corrupting it
    to '=', '/', an em-dash, ...) + up to a few junk characters + a digit. Requiring
    the separator specifically (not just "any non-alnum") matters here: single-letter
    prefixes like "F" and "E" otherwise collide with the "°F" temperature unit.
    Returns (position, prefix) pairs sorted left-to-right."""
    hits = []
    for prefix in prefixes:
        match = re.search(rf"\b{re.escape(prefix)}[-=/–—][^\n]{{0,3}}\d", text)
        if match:
            hits.append((match.start(), prefix))
    return sorted(hits)


def _ocr_header(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    x0, x1 = int(w * HEADER_MARGIN_FRACTION), int(w * (1 - HEADER_MARGIN_FRACTION))
    header = image[: int(h * HEADER_HEIGHT_FRACTION), x0:x1]
    gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY) if header.ndim == 3 else header
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(binary, config="--psm 6")


def extract_title_block_limits(image: np.ndarray, known_tags: set[str] | None = None) -> dict[str, dict]:
    """Best-effort: OCR a page's header block, pair each tag found there with the
    design pressure/temperature clause at the same left-to-right position."""
    text = _ocr_header(image).upper()

    if known_tags:
        # Real bug found and fixed: a plain {prefix: tag} dict silently dropped
        # every known tag but one whenever two known tags shared a prefix letter
        # (e.g. two "MV-" equipment tags) -- that tag became permanently
        # unmatchable via title-block OCR with no signal it happened. Grouping
        # into a list means every tag sharing an ambiguous prefix now gets
        # surfaced (with the same clause, since OCR genuinely can't tell them
        # apart from a single header instance) instead of one winning silently.
        known_prefixes: dict[str, list[str]] = {}
        for tag in known_tags:
            known_prefixes.setdefault(tag.split("-")[0], []).append(tag)
        prefix_hits = _prefix_positions(text, set(known_prefixes))
    else:
        # No vocabulary supplied — best effort, strict prefix+digit pattern only.
        prefix_hits = [(m.start(), m.group(1)) for m in PREFIX_PATTERN.finditer(text)]
        known_prefixes = {prefix: [prefix] for _, prefix in prefix_hits}

    if not prefix_hits:
        return {}

    pressures = [(m.start(), float(m.group(1))) for m in PRESSURE_PATTERN.finditer(text)]
    temperatures = [(m.start(), m.group(1)) for m in TEMPERATURE_PATTERN.finditer(text)]

    results = {}
    for i, (_, prefix) in enumerate(prefix_hits):
        # Real bug found and fixed: falling back to pressures[0]/temperatures[0]
        # when OCR found fewer clauses than tags silently misattributed one
        # equipment's design limit to another instead of leaving it unresolved.
        # None is the honest result when the clause count doesn't line up.
        pressure = pressures[i][1] if i < len(pressures) else None
        temperature = temperatures[i][1] if i < len(temperatures) else None
        for tag in known_prefixes[prefix]:
            results[tag] = {"design_pressure_psig": pressure, "design_temperature_f": temperature, "raw_header": text}
    return results


def extract_title_block_limits_all_pages(
    pdf_path: str | Path, known_tags: set[str] | None = None, dpi: int = 300
) -> dict[str, dict]:
    """Run extract_title_block_limits over every sheet and merge results."""
    results: dict[str, dict] = {}
    for image in pdf_to_images(pdf_path, dpi=dpi):
        results.update(extract_title_block_limits(image, known_tags))
    return results
