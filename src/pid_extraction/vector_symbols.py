"""Detect instrument-bubble-style circle symbols from PDF vector path data.

Time-boxed follow-on to vector_lines.py (see README "Vector-Based Symbol
Detection"). Raster contour detection undercounts real ISA symbols badly (see
README "Component-graph extraction: honest results, and why"); this recovers
one specific, well-defined symbol class — small near-square closed vector
paths — directly from exact path geometry instead.

Calibration note: on page 0 of the real assignment PDF, small near-square
closed paths cluster tightly around ~17pt (confirmed against a symbol visually
identified as an instrument bubble). A separate ~33pt cluster looked promising
at first but turned out to be the page border's corner tick brackets (all 3
hits sat at the page's own corners) — excluded, not a symbol class.
SYMBOL_SIZE_BANDS_PT encodes the surviving empirical band; it is specific to
this drawing set's stencil/font scale, not a general ISA constant — a
different P&ID would need recalibrating it the same way (dump a size
histogram of near-square closed paths, look for the cluster, sanity-check
hits aren't sitting on the border).
"""
from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.pid_extraction.shape_detection import DetectedShape

MIN_ASPECT, MAX_ASPECT = 0.7, 1.4
# (min_pt, max_pt) bands, calibrated as described above.
SYMBOL_SIZE_BANDS_PT = [(14.0, 20.0)]


def _in_any_band(value: float, bands: list[tuple[float, float]]) -> bool:
    return any(lo <= value <= hi for lo, hi in bands)


def extract_circle_symbols(pdf_path: str | Path, page_index: int = 0, dpi: int = 300) -> list[DetectedShape]:
    """Return candidate instrument-bubble symbols as DetectedShape objects, in the
    same pixel space pdf_to_images(pdf_path, dpi=dpi) renders into — usable
    anywhere a raster-detected DetectedShape is (OCR tagging, connector matching,
    graph building)."""
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
            r = d["rect"]
            if r.width <= 0 or r.height <= 0:
                continue
            aspect = r.width / r.height
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            if not _in_any_band(max(r.width, r.height), SYMBOL_SIZE_BANDS_PT):
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
                    kind="circle",
                    bbox=bbox,
                    center=(int((x0 + x1) / 2), int((y0 + y1) / 2)),
                    contour=np.empty((0, 1, 2), dtype=np.int32),
                )
            )
        return shapes
