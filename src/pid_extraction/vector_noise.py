"""Identify decorative "coil/tubing" symbol regions to exclude from other
detectors' candidate searches.

This P&ID uses a scalloped cloud/coil glyph (labeled "TUBING" on the real
sheet) wherever flexible tubing is shown. Each scallop bump is its own tiny
2-item bezier-curve path (~2-9pt), and a full coil symbol is a chain of
10-16 of these evenly spaced around its circumference — confirmed by
rendering one such region and seeing the actual "TUBING" cloud glyph.

Why this exists: these small curve fragments were inflating candidate counts
for other exploratory detection (e.g. checking pipe-segment endpoints for
missed valves — see README "Closing the Valve Gap Further") without being a
symbol of interest themselves. Detecting and excluding the whole coil region
is more reliable than trying to filter each tiny arc's endpoints individually.
"""
from __future__ import annotations

from pathlib import Path

import fitz

_ARC_MIN_PT, _ARC_MAX_PT = 1.5, 10.0
_CLUSTER_GAP_PT = 6.0
_MIN_ARCS_PER_COIL = 8


def _is_coil_arc(d: dict) -> bool:
    if {it[0] for it in d["items"]} != {"c"}:
        return False
    if len(d["items"]) != 2:
        return False
    r = d["rect"]
    if r.width <= 0 or r.height <= 0:
        return False
    long_dim, short_dim = max(r.width, r.height), min(r.width, r.height)
    return _ARC_MIN_PT <= short_dim <= _ARC_MAX_PT and _ARC_MIN_PT <= long_dim <= 3 * _ARC_MAX_PT


def find_coil_regions_mediabox(pdf_path: str | Path, page_index: int = 0) -> list[fitz.Rect]:
    """Return mediabox-space bounding rects of detected coil/tubing symbols
    (each the union of >= _MIN_ARCS_PER_COIL nearby arc primitives)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"P&ID PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        if page_index >= doc.page_count:
            raise ValueError(f"Requested page {page_index}, PDF only has {doc.page_count} page(s)")
        page = doc[page_index]

        arcs = [d["rect"] for d in page.get_drawings() if d["type"] in ("s", "fs") and _is_coil_arc(d)]

    n = len(arcs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def close(r1: fitz.Rect, r2: fitz.Rect, gap: float) -> bool:
        return not (r1.x1 + gap < r2.x0 or r2.x1 + gap < r1.x0 or r1.y1 + gap < r2.y0 or r2.y1 + gap < r1.y0)

    for i in range(n):
        for j in range(i + 1, n):
            if close(arcs[i], arcs[j], _CLUSTER_GAP_PT):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    regions = []
    for idxs in clusters.values():
        if len(idxs) < _MIN_ARCS_PER_COIL:
            continue
        xs0 = [arcs[i].x0 for i in idxs]
        ys0 = [arcs[i].y0 for i in idxs]
        xs1 = [arcs[i].x1 for i in idxs]
        ys1 = [arcs[i].y1 for i in idxs]
        regions.append(fitz.Rect(min(xs0), min(ys0), max(xs1), max(ys1)))
    return regions
