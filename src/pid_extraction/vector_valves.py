"""Detect valve bowtie symbols from PDF vector path data.

Two signatures, found and validated separately — see README "Vector-Based
Valve Detection" for the full account of what was tried and ruled out first
(a broadened isolated-shape search found nothing new; open spatial clustering
chained across unrelated regions).

**Isolated bowtie** (`_ISOLATED_*` constants): a valve drawn as its own closed
all-line path, ~20-30 short segments, near-square 9-14pt bounding box. Found
by rendering one candidate at 600 DPI and matching it against a bowtie next to
an instrument connection.

**Fused pipe+valve** (`_FUSED_*` constants): most valves on this drawing are
NOT isolated shapes — they're tessellated as part of the same continuous
vector path as their connecting pipe run (confirmed: rendering
`MV-715-02A`'s candidate showed the full pipe segment, arrowhead, and valve
bowtie as one 12-line-item path). Found by looking for elongated all-line
paths with a *moderate* short dimension (5-25pt — thicker than a bare pipe
stroke's ~0.5-2.5pt, because the bowtie's width dominates the bounding box at
whatever point it sits on the run) and 12 line items specifically. Checked
against 4 real candidates matching this signature: 3 were confirmed valves
(`MV-715-10A`, `MV-715-10B`, one more), 1 was a false positive (an off-page
destination reference box, not a valve) — ~75% precision, not perfect,
reported honestly rather than silently accepted. A *different* nearby
item-count (7 items, same size range) was checked too and turned out to be an
unrelated symbol (a drawing cross-reference flag near the row-number ladder)
— excluded, not included on a guess.

Known imprecision: a fused-signature match's bbox is the *entire pipe run's*
bounding box, not a tight crop around just the valve bowtie (which typically
sits at one end of the run, not its center) — so OCR tag-reading against
these specific detections is expected to be less reliable than for the
isolated-bowtie signature. Not corrected here; disclosed instead.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.pid_extraction.shape_detection import DetectedShape
from src.pid_extraction.vector_lines import _to_pixel

_ISOLATED_MIN_SIDE_PT, _ISOLATED_MAX_SIDE_PT = 9.0, 14.0
_ISOLATED_MIN_ASPECT, _ISOLATED_MAX_ASPECT = 0.7, 1.4
_ISOLATED_MIN_ITEMS, _ISOLATED_MAX_ITEMS = 20, 30

_FUSED_ITEMS = 12
_FUSED_MIN_LONG_PT, _FUSED_MAX_LONG_PT = 15.0, 250.0
_FUSED_MIN_SHORT_PT, _FUSED_MAX_SHORT_PT = 5.0, 25.0


def _is_isolated_bowtie(d: dict) -> bool:
    if {it[0] for it in d["items"]} != {"l"}:
        return False
    if not (_ISOLATED_MIN_ITEMS <= len(d["items"]) <= _ISOLATED_MAX_ITEMS):
        return False
    r = d["rect"]
    if r.width <= 0 or r.height <= 0:
        return False
    aspect = r.width / r.height
    if not (_ISOLATED_MIN_ASPECT <= aspect <= _ISOLATED_MAX_ASPECT):
        return False
    return _ISOLATED_MIN_SIDE_PT <= r.width <= _ISOLATED_MAX_SIDE_PT and _ISOLATED_MIN_SIDE_PT <= r.height <= _ISOLATED_MAX_SIDE_PT


def _is_fused_pipe_valve(d: dict) -> bool:
    if {it[0] for it in d["items"]} != {"l"}:
        return False
    if len(d["items"]) != _FUSED_ITEMS:
        return False
    r = d["rect"]
    if r.width <= 0 or r.height <= 0:
        return False
    long_dim, short_dim = max(r.width, r.height), min(r.width, r.height)
    return _FUSED_MIN_LONG_PT <= long_dim <= _FUSED_MAX_LONG_PT and _FUSED_MIN_SHORT_PT <= short_dim <= _FUSED_MAX_SHORT_PT


def extract_valve_symbols(pdf_path: str | Path, page_index: int = 0, dpi: int = 300) -> list[DetectedShape]:
    """Return candidate valve symbols (both signatures) as DetectedShape objects,
    in the same pixel space pdf_to_images(pdf_path, dpi=dpi) renders into."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"P&ID PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        if page_index >= doc.page_count:
            raise ValueError(f"Requested page {page_index}, PDF only has {doc.page_count} page(s)")
        page = doc[page_index]
        rotation_matrix = page.rotation_matrix

        shapes = []
        for d in page.get_drawings():
            if d["type"] not in ("s", "fs"):
                continue
            if not (_is_isolated_bowtie(d) or _is_fused_pipe_valve(d)):
                continue

            r = d["rect"]
            corners = [
                _to_pixel(fitz.Point(r.x0, r.y0), rotation_matrix, dpi),
                _to_pixel(fitz.Point(r.x1, r.y1), rotation_matrix, dpi),
            ]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
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
