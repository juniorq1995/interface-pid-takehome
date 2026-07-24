"""Classical-CV component detection for P&ID diagrams.

No labeled P&ID symbol dataset is available for this assignment, so this uses
contour/shape heuristics (OpenCV) rather than a trained detector (YOLO etc).
See README "Approach & Assumptions" for the documented tradeoff and the path
to swapping in a fine-tuned detector later without touching downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_CONTOUR_AREA = 600
# Larger than connector line thickness, smaller than the smallest symbol stroke —
# opening severs thin pipe lines from touching symbol bodies before contouring,
# and also erases stray text-glyph blobs that would otherwise be misread as symbols.
OPENING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


@dataclass
class DetectedShape:
    shape_id: int
    kind: str  # "circle" | "rectangle" | "bowtie" | "circle_triangle" | "unknown"
    bbox: tuple[int, int, int, int]  # x, y, w, h
    center: tuple[int, int]
    contour: np.ndarray = field(repr=False)


def _classify(contour: np.ndarray, child_contours: list[np.ndarray]) -> str:
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return "unknown"

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "unknown"

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    circularity = 4 * np.pi * area / (perimeter**2)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / h if h else 0

    for child in child_contours:
        child_area = cv2.contourArea(child)
        if child_area == 0:
            continue
        ratio = child_area / area
        child_approx = cv2.approxPolyDP(child, 0.02 * cv2.arcLength(child, True), True)
        if 0.05 < ratio < 0.6 and 3 <= len(child_approx) <= 4:
            return "circle_triangle"  # pump: circle housing + impeller triangle

    # Rectangle check comes first: a square's circularity (~0.78) already clears the
    # "circle" threshold below, so vertex count (not circularity) is the discriminator.
    if len(approx) == 4 and solidity > 0.9 and 0.5 < aspect < 2.0:
        return "rectangle"  # tank/vessel
    if len(approx) >= 6 and circularity > 0.75 and solidity > 0.85:
        return "circle"  # instrument bubble
    if circularity < 0.7 and 4 <= len(approx) <= 8:
        return "bowtie"  # valve (notched/concave glyph — low circularity from the indentation)
    return "unknown"


def detect_shapes(image: np.ndarray) -> list[DetectedShape]:
    """Detect P&ID component symbols in a rendered diagram image (BGR or grayscale)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, OPENING_KERNEL)

    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    children_by_parent: dict[int, list[np.ndarray]] = {}
    for idx, h in enumerate(hierarchy):
        parent = h[3]
        if parent != -1:
            children_by_parent.setdefault(parent, []).append(contours[idx])

    shapes: list[DetectedShape] = []
    for idx, contour in enumerate(contours):
        # Only classify top-level (outer) contours; children are consumed as pump impellers.
        if hierarchy[idx][3] != -1:
            continue
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        kind = _classify(contour, children_by_parent.get(idx, []))
        if kind == "unknown":
            continue
        x, y, w, h = cv2.boundingRect(contour)
        shapes.append(
            DetectedShape(
                shape_id=len(shapes),
                kind=kind,
                bbox=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                contour=contour,
            )
        )
    return shapes
