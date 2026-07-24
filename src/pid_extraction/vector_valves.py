"""Detect valve bowtie symbols from PDF vector path data.

Same approach and rationale as vector_symbols.py's instrument-bubble detector,
applied to a different signature: a valve bowtie is drawn as an all-line closed
path (no bezier curves), tessellated into ~20-30 short segments by this CAD
export, in a near-square bounding box roughly 9-14pt per side.

Calibration note: confirmed by visually rendering one candidate at 600 DPI and
matching it against a bowtie symbol next to an instrument connection (see
README "Vector-Based Valve Detection"). Unlike instrument bubbles — which are
uniform circles regardless of orientation — valve bowties can be drawn rotated
(horizontal vs. vertical pipe runs), which changes their tessellation and is
the known reason this detector's recall is partial, not exhaustive. Documented
honestly in README rather than claimed as solved.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.pid_extraction.shape_detection import DetectedShape

MIN_SIDE_PT, MAX_SIDE_PT = 9.0, 14.0
MIN_ASPECT, MAX_ASPECT = 0.7, 1.4
MIN_LINE_ITEMS, MAX_LINE_ITEMS = 20, 30


def extract_valve_symbols(pdf_path: str | Path, page_index: int = 0, dpi: int = 300) -> list[DetectedShape]:
    """Return candidate valve-bowtie symbols as DetectedShape objects, in the
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
            if not (MIN_LINE_ITEMS <= len(d["items"]) <= MAX_LINE_ITEMS):
                continue

            r = d["rect"]
            if r.width <= 0 or r.height <= 0:
                continue
            aspect = r.width / r.height
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            if not (MIN_SIDE_PT <= r.width <= MAX_SIDE_PT and MIN_SIDE_PT <= r.height <= MAX_SIDE_PT):
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
                    kind="bowtie",
                    bbox=bbox,
                    center=(int((x0 + x1) / 2), int((y0 + y1) / 2)),
                    contour=np.empty((0, 1, 2), dtype=np.int32),
                )
            )
        return shapes
