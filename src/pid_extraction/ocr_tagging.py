"""OCR-based tag extraction for detected P&ID component symbols."""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract

from src.component_types import TAG_PATTERN
from src.pid_extraction.shape_detection import DetectedShape


def _crop_label_region(image: np.ndarray, shape: DetectedShape, margin: int = 40) -> np.ndarray:
    x, y, w, h = shape.bbox
    ih, iw = image.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin // 2)
    x1 = min(iw, x + w + margin)
    y1 = min(ih, y + h + margin * 2)  # tags are usually placed below the symbol
    return image[y0:y1, x0:x1]


def extract_tag(image: np.ndarray, shape: DetectedShape) -> tuple[str | None, str]:
    """Return (normalized_tag_or_None, raw_ocr_text) for a detected shape."""
    crop = _crop_label_region(image, shape)
    if crop.size == 0:
        return None, ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    raw_text = pytesseract.image_to_string(
        binary, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    ).strip()

    match = TAG_PATTERN.search(raw_text.upper())
    return (match.group(1) if match else None), raw_text
