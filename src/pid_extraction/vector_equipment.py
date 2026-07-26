"""Detect equipment vessel outlines (tanks/filters/towers) from PDF vector path data.

Same recipe as vector_symbols.py and vector_valves.py: rendered and visually
confirmed a candidate before trusting it, then checked it generalizes.

Calibration note, including two real mistakes made and caught along the way
(see README "Vector-Based Equipment Detection" for the full account) — read
before extending this to a different document:

1. A generic size/aspect heuristic's top hit was the title-block logo, not a
   vessel. Caught by rendering, dropped.
2. A single-path search anchored off a known valve found a candidate that
   *looked* confirmed in a generously-padded render crop — but the padding was
   small relative to the candidate's own size, so the crop was dominated by
   whatever was nearby, not the target shape itself. Re-checked against the
   *exact* unpadded bbox: it was an unrelated horizontal pipe stroke.
3. Spatial clustering of primitives (the technique the research literature
   actually recommends for compound symbols) found the right neighborhood but
   suffered single-linkage "chaining" — bridging transitively across unrelated
   regions once the gap tolerance was opened enough to bridge this document's
   dashed level-indicator lines.

What actually worked: this vessel type turned out to be drawn as a *single*
compound stroked path (one `s`-type drawing, ~74-78 line segments, all type
`l`) encoding the flange, cylindrical body, dashed level lines, and domed head
together — not a compound of separate primitives needing clustering at all.
Found by searching for line-item-count clusters directly (60-90 items) in a
plausible vessel-scale bounding box (80-130 x 15-35pt), confirmed against BOTH
real vessels on page 0 by rendering each with minimal (8pt) padding — tight
enough that padding couldn't dominate the frame the way mistake #2 above did.

Scope: validated against this document's cylindrical-filter-vessel style
(F-715A/B) only. Other equipment types on this document (V-745 stabilizer
tower, E-742 exchanger, AC-746 after-cooler) were not checked and may use a
different vector structure entirely — not claimed to generalize past what was
actually confirmed.

Second signature added later (page 1's V-745 stabilizer tower): unlike
F-715's single all-encompassing compound path, V-745's cylindrical body turned
out NOT to be one unified stroke at all — searching page 1 for any compound
line-only path above 30 items found only one real candidate (48 items), far
separated from the next-largest unrelated cluster (37 items). Rendered it
tightly and confirmed: it's the vessel's domed top + mesh-pad, not the full
body — but that's sufficient to register the vessel's presence for the
category-coverage metric this project scores against (count-based, not
bbox-precise). Confirmed this signature (40-55 items, 8-20pt wide, 50-80pt
tall — a tall/narrow footprint, the opposite orientation of F-715's wide/short
one) produces zero false positives on pages 0 and 2. The full vessel body
itself is presumably built from many small separate primitives rather than one
compound path — not pursued further; clustering was already rejected for
F-715 due to chaining, and this fragment-based match is enough to close the
coverage gap without that risk.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.pid_extraction.shape_detection import DetectedShape

MIN_ITEMS, MAX_ITEMS = 60, 90
MIN_WIDTH_PT, MAX_WIDTH_PT = 80.0, 130.0
MIN_HEIGHT_PT, MAX_HEIGHT_PT = 15.0, 35.0

# V-745's domed-top + mesh-pad fragment (see module docstring) — a distinct,
# non-overlapping numeric range from the F-715 signature above: fewer items,
# tall/narrow instead of wide/short.
TOWER_TOP_MIN_ITEMS, TOWER_TOP_MAX_ITEMS = 40, 55
TOWER_TOP_MIN_WIDTH_PT, TOWER_TOP_MAX_WIDTH_PT = 8.0, 20.0
TOWER_TOP_MIN_HEIGHT_PT, TOWER_TOP_MAX_HEIGHT_PT = 50.0, 80.0


def _matches_any_vessel_signature(item_count: int, width_pt: float, height_pt: float) -> bool:
    if MIN_ITEMS <= item_count <= MAX_ITEMS and MIN_WIDTH_PT <= width_pt <= MAX_WIDTH_PT and MIN_HEIGHT_PT <= height_pt <= MAX_HEIGHT_PT:
        return True
    return (
        TOWER_TOP_MIN_ITEMS <= item_count <= TOWER_TOP_MAX_ITEMS
        and TOWER_TOP_MIN_WIDTH_PT <= width_pt <= TOWER_TOP_MAX_WIDTH_PT
        and TOWER_TOP_MIN_HEIGHT_PT <= height_pt <= TOWER_TOP_MAX_HEIGHT_PT
    )


def extract_vessel_symbols(pdf_path: str | Path, page_index: int = 0, dpi: int = 300) -> list[DetectedShape]:
    """Return candidate vessel-outline symbols as DetectedShape objects, in the
    same pixel space pdf_to_images(pdf_path, dpi=dpi) renders into."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"P&ID PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        if page_index >= doc.page_count:
            raise ValueError(f"Requested page {page_index}, PDF only has {doc.page_count} page(s)")
        page = doc[page_index]
        rotation_matrix = page.rotation_matrix
        scale = dpi / 72

        shapes = []
        for d in page.get_drawings():
            if d["type"] not in ("s", "fs"):
                continue
            item_types = {it[0] for it in d["items"]}
            if item_types != {"l"}:
                continue

            r = d["rect"]
            if r.width <= 0 or r.height <= 0:
                continue
            if not _matches_any_vessel_signature(len(d["items"]), r.width, r.height):
                continue

            corners = [fitz.Point(r.x0, r.y0) * rotation_matrix, fitz.Point(r.x1, r.y1) * rotation_matrix]
            xs = [p.x * scale for p in corners]
            ys = [p.y * scale for p in corners]
            x0, x1 = sorted(xs)
            y0, y1 = sorted(ys)
            bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            shapes.append(
                DetectedShape(
                    shape_id=len(shapes),
                    kind="rectangle",  # reuses the existing "tank" mapping in component_types.SHAPE_TO_TYPE
                    bbox=bbox,
                    center=(int((x0 + x1) / 2), int((y0 + y1) / 2)),
                    contour=np.empty((0, 1, 2), dtype=np.int32),
                )
            )
        return shapes
