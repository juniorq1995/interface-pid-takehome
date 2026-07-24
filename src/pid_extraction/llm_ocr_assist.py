"""LLM-assisted tag reading, as a fallback when Tesseract OCR fails.

Uses a local Ollama vision model (llama3.2-vision) to read a symbol's label
crop directly from the image, rather than post-hoc correcting Tesseract's
already-mangled text output. On the real Interface P&ID, tiny instrument-bubble
crops (~50-70px) produced empty or garbled Tesseract reads (see README) — this
gives the model the actual pixels instead of asking it to guess from noise.

Local/open-source by design (Ollama, not a hosted API) — no data leaves the
machine, no API key, no per-call cost. Opt-in and additive: only invoked when
the existing deterministic OCR path (ocr_tagging.py) comes back empty, so the
fast/free/deterministic path is always tried first.
"""
from __future__ import annotations

import base64
import re

import cv2
import numpy as np
import requests

from src.component_types import TAG_CORE
from src.pid_extraction.ocr_tagging import _crop_label_region
from src.pid_extraction.shape_detection import DetectedShape

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llama3.2-vision:11b"
REQUEST_TIMEOUT_S = 90

LOOSE_TAG_PATTERN = re.compile(rf"\b({TAG_CORE}|[A-Z]{{1,4}}\s+\d{{2,4}}[A-Z]?)\b")

PROMPT = (
    "This is a cropped region from an industrial P&ID (piping and instrumentation "
    "diagram). It contains one small instrument or equipment symbol with an adjacent "
    "text tag, e.g. 'PI 715A' or 'FT-101'. Reply with ONLY the tag text you can read, "
    "exactly as printed, nothing else. If you cannot confidently read a tag, reply "
    "with exactly: UNKNOWN"
)


def _encode_crop(crop: np.ndarray) -> str:
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", scaled)
    if not ok:
        raise ValueError("Failed to encode crop for LLM OCR assist")
    return base64.b64encode(buf).decode("ascii")


def ollama_available() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def read_tag_with_llm(image: np.ndarray, shape: DetectedShape) -> tuple[str | None, str]:
    """Return (normalized_tag_or_None, raw_model_text). Best-effort: any network/
    model failure degrades to (None, "") rather than raising, since this is an
    optional enhancement layered over the deterministic OCR path."""
    crop = _crop_label_region(image, shape)
    if crop.size == 0:
        return None, ""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": PROMPT,
                "images": [_encode_crop(crop)],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
    except (requests.exceptions.RequestException, ValueError):
        return None, ""

    if raw_text.upper() == "UNKNOWN":
        return None, raw_text

    match = LOOSE_TAG_PATTERN.search(raw_text.upper())
    if not match:
        return None, raw_text
    return re.sub(r"\s+", " ", match.group(1)), raw_text
