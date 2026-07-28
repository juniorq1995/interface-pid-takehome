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


# Real gap found by direct inspection of failed valve-tag crops (not
# theorized): most unread valve tags on this document are drawn rotated 90°
# alongside vertical pipe runs (e.g. "MV-715-04B" reads bottom-to-top), and
# the original single-attempt `--psm 7` call assumes horizontal text --
# structurally can't read these regardless of image quality. Both cheap,
# local, deterministic fixes (no new dependency: cv2.rotate is already
# available) are tried per crop before falling through to the LLM-assist
# fallback in pipeline.py, so any tag recovered here is free compared to a
# 30-90s LLM call.
_ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
_PSM_CONFIGS = ("--psm 7", "--psm 11")  # single line, then sparse-text fallback


def _ocr_once(binary_crop: np.ndarray, psm_config: str) -> tuple[str | None, str]:
    raw_text = pytesseract.image_to_string(
        binary_crop, config=f"{psm_config} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    ).strip()
    match = TAG_PATTERN.search(raw_text.upper())
    return (match.group(1) if match else None), raw_text


def extract_tag(image: np.ndarray, shape: DetectedShape) -> tuple[str | None, str]:
    """Return (normalized_tag_or_None, raw_ocr_text) for a detected shape.

    Tries the crop at 0°/90°CW/90°CCW, each at two PSM configs, stopping at
    the first rotation+PSM combo that yields a valid tag match. `raw_text`
    reflects the last attempt made (or the winning one, if a match was
    found) -- always the most recent OCR read, useful for debugging even on
    a full miss.
    """
    crop = _crop_label_region(image, shape)
    if crop.size == 0:
        return None, ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    last_raw_text = ""
    for rotation in _ROTATIONS:
        rotated = binary if rotation is None else cv2.rotate(binary, rotation)
        for psm_config in _PSM_CONFIGS:
            tag, raw_text = _ocr_once(rotated, psm_config)
            if raw_text:
                last_raw_text = raw_text
            if tag:
                return tag, raw_text
    return None, last_raw_text
