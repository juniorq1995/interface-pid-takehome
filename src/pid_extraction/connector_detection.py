"""Pipe/connector detection between P&ID symbols.

Approach: mask out symbol interiors, run probabilistic Hough line detection on
what's left (the pipe strokes only), then attach each detected segment's two
endpoints to whichever symbol bbox they land nearest. This avoids the false
positive of three inline, individually-connected symbols (A-B, B-C) reading as
a spurious direct A-C edge, which a naive "ink continuity between centers"
check would produce.

Assumption (documented in README): connections are drawn as straight-line
segments. Curved or multi-bend orthogonal pipe runs are a known limitation.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.pid_extraction.shape_detection import DetectedShape

HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10
ENDPOINT_MAX_DIST = 20


def _binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _mask_symbol_interiors(binary: np.ndarray, shapes: list[DetectedShape], pad: int = 5) -> np.ndarray:
    masked = binary.copy()
    ih, iw = masked.shape[:2]
    for shape in shapes:
        x, y, w, h = shape.bbox
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(iw, x + w + pad), min(ih, y + h + pad)
        masked[y0:y1, x0:x1] = 0
    return masked


def _bbox_distance(bbox: tuple[int, int, int, int], point: tuple[float, float]) -> float:
    x, y, w, h = bbox
    px, py = point
    dx = max(x - px, 0, px - (x + w))
    dy = max(y - py, 0, py - (y + h))
    return float(np.hypot(dx, dy))


def _nearest_shape(shapes: list[DetectedShape], point: tuple[float, float], max_dist: float = ENDPOINT_MAX_DIST) -> DetectedShape | None:
    best, best_dist = None, max_dist
    for shape in shapes:
        dist = _bbox_distance(shape.bbox, point)
        if dist < best_dist:
            best, best_dist = shape, dist
    return best


def _edges_from_segments(
    segments, shapes: list[DetectedShape], max_dist: float = ENDPOINT_MAX_DIST
) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for (x1, y1), (x2, y2) in segments:
        shape_a = _nearest_shape(shapes, (x1, y1), max_dist)
        shape_b = _nearest_shape(shapes, (x2, y2), max_dist)
        if shape_a is not None and shape_b is not None and shape_a.shape_id != shape_b.shape_id:
            edges.add(tuple(sorted((shape_a.shape_id, shape_b.shape_id))))
    return sorted(edges)


def detect_connections(image: np.ndarray, shapes: list[DetectedShape]) -> list[tuple[int, int]]:
    """Raster fallback: mask symbol interiors, Hough-detect lines in what's left,
    attach each segment's endpoints to the nearest symbol. Used when the PDF has
    no vector path data (e.g. a scanned P&ID) — see vector_lines.py for the
    exact-geometry path used when vector data is available."""
    if len(shapes) < 2:
        return []

    binary = _binarize(image)
    line_mask = _mask_symbol_interiors(binary, shapes)

    hough_segments = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if hough_segments is None:
        return []

    # cv2.HoughLinesP's output ndim varies by OpenCV version: (N, 1, 4) on older
    # builds, (N, 4) as of opencv 5.0 — reshape rather than assume either.
    segments = [((x1, y1), (x2, y2)) for (x1, y1, x2, y2) in hough_segments.reshape(-1, 4)]
    return _edges_from_segments(segments, shapes)


def detect_connections_from_vector_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]], shapes: list[DetectedShape]
) -> list[tuple[int, int]]:
    """Same endpoint-to-symbol attachment as detect_connections, but for exact
    vector-derived segments (src/pid_extraction/vector_lines.py) instead of
    Hough-detected raster ones. No image/masking needed — the segments are exact."""
    if len(shapes) < 2 or not segments:
        return []
    return _edges_from_segments(segments, shapes)
