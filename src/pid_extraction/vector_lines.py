"""Extract pipe/connector line segments from a PDF's native vector path data,
when available, instead of re-deriving line geometry from a rasterized image.

Why: on the real Interface P&ID, raster Hough-line detection has to fight
rasterization noise and a topologically-fused sheet (border + grid + pipes +
symbols all touching). The PDF underneath is CAD-exported and carries exact
vector geometry (thousands of stroked path objects per sheet) — reading pipe
segments directly from that sidesteps rasterization entirely and is exact,
not inferred. See README "Vector-Based Connector Detection" for the full story,
including why this is scoped to *connectors* and not full symbol recognition.

Coordinate handling: PyMuPDF returns drawing coordinates in the page's
un-rotated MediaBox space. This document's pages are stored rotated 270°
(portrait MediaBox, landscape display) — `page.rotation_matrix` maps MediaBox
points into the same display-space coordinate system `pdf_to_images` renders
into, and multiplying by dpi/72 lands in that render's pixel space.
"""
from __future__ import annotations

from pathlib import Path

import fitz

MIN_LENGTH_PT = 8.0
MAX_THICKNESS_PT = 2.5
# A stroked drawing this long relative to the page is almost certainly the sheet
# border/frame, not a pipe run — no pipe on this document spans that much of the
# sheet. Relative (not an absolute point value) so it isn't tuned to one page size.
MAX_LENGTH_FRACTION_OF_PAGE = 0.5


def _to_pixel(point: fitz.Point, rotation_matrix: fitz.Matrix, dpi: int) -> tuple[float, float]:
    transformed = point * rotation_matrix
    scale = dpi / 72
    return (transformed.x * scale, transformed.y * scale)


def extract_line_segments(
    pdf_path: str | Path, page_index: int = 0, dpi: int = 300
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return pipe-like line segments as ((x1,y1),(x2,y2)) in the same pixel space
    pdf_to_images(pdf_path, dpi=dpi) renders into. Empty list if the PDF has no
    vector drawing data (e.g. a scanned/rasterized P&ID) — callers should fall
    back to raster connector detection in that case."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"P&ID PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        if page_index >= doc.page_count:
            raise ValueError(f"Requested page {page_index}, PDF only has {doc.page_count} page(s)")
        page = doc[page_index]
        rotation_matrix = page.rotation_matrix
        max_length = MAX_LENGTH_FRACTION_OF_PAGE * min(page.rect.width, page.rect.height)

        segments = []
        for d in page.get_drawings():
            if d["type"] not in ("s", "fs"):
                continue
            r = d["rect"]
            long_dim, short_dim = max(r.width, r.height), min(r.width, r.height)
            if not (MIN_LENGTH_PT <= long_dim <= max_length and short_dim <= MAX_THICKNESS_PT):
                continue
            # Approximate the stroked path by its bounding box's long diagonal —
            # accurate for the near-horizontal/near-vertical runs typical of P&ID
            # pipe routing, which is the case this is scoped for (see README).
            p1 = _to_pixel(fitz.Point(r.x0, r.y0), rotation_matrix, dpi)
            p2 = _to_pixel(fitz.Point(r.x1, r.y1), rotation_matrix, dpi)
            segments.append((p1, p2))

        return segments
