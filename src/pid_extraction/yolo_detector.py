"""Trained-model symbol detection: a YOLOv8 model fine-tuned on the Roboflow
"P&ID Symbols R2" dataset (2,742 real images, 180 labeled classes — see
datasets/pid-symbols-r2/, staged earlier this session as R&D material).

This is the detector every heuristic vector-geometry attempt this session
converged on needing (see README "Closing the Valve Gap Further" and
"Vector-Based Valve Detection") — confirmed by an independent open-source
project ([Paradox-85/PidDetector]) using the identical approach as its
production method. Unlike the vector-path heuristics elsewhere in this
package, this detector works directly on the rendered raster image and
doesn't depend on the PDF having usable vector data at all.

Class names in the dataset are specific ISA subtypes ("Gate Valve", "Hand
Operated Ball Valve", ...) — COMPONENT_TYPE_KEYWORDS maps them to this
project's coarser component_type vocabulary by keyword, since exact 1:1
mapping to component_types.py's existing categories isn't meaningful (that
vocabulary was built from ISA tag prefixes, not this dataset's class taxonomy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pid_extraction.shape_detection import DetectedShape

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# ultralytics ignores a relative `project=` override and nests it under its own
# `runs/detect/` default instead of using it as the run root directly — the
# weights actually landed at runs/detect/datasets/runs/..., not datasets/runs/...
# as the training command's `project=datasets/runs` argument implied.
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "runs" / "detect" / "datasets" / "runs" / "pid_coarse_v1" / "weights" / "best.pt"
DEFAULT_CONFIDENCE = 0.25

# The training set is Roboflow's own pre-cropped 640x640 tiles; our real P&ID
# page renders at 3300x5100 (300 dpi). Feeding the whole page to the model in
# one pass resizes it down to fit imgsz=640, shrinking valve-sized symbols to
# a handful of pixels — confirmed by direct measurement: 5 raw detections on
# the full page at conf=0.15, vs 19 raw (pre-merge) detections tiling the same
# page into 640px crops. Tiling is only applied when the image is actually
# larger than one tile — small test fixtures take the original single-pass path.
TILE_SIZE = 640
TILE_OVERLAP = 96
MERGE_IOU_THRESHOLD = 0.5

# Longest/most-specific keyword first within each category isn't required here —
# checked as substring membership, category assignment is first-match-wins below.
_KEYWORD_TO_TYPE = {
    "valve": "valve",
    "psv": "safety_valve",
    "prv": "relief_valve",
    "pump": "pump",
    "compressor": "compressor",
    "blower": "compressor",
    "turbine": "turbine",
    "exchanger": "heat_exchanger",
    "tank": "tank",
    "vessel": "tank",
    "tower": "tank",
    "drum": "tank",
    "reactor": "tank",
    "indicator": "instrument",
    "transmitter": "instrument",
    "controller": "instrument",
    "recorder": "instrument",
    "gauge": "instrument",
    "alarm": "instrument",
    "meter": "instrument",
    "rotameter": "instrument",
}


_COARSE_CLASS_NAMES = {"valve", "instrument", "equipment"}


def class_name_to_component_type(class_name: str) -> str:
    """Two different model taxonomies get fed through here: Roboflow's ~180
    fine-grained ISA subtype names ("Gate Valve", "Pressure Transmitter", ...),
    resolved by keyword substring below, and the merged coarse-class model's
    exact 3 category names ("valve"/"instrument"/"equipment") — those must be
    returned directly, since "instrument" and "equipment" don't contain any of
    the fine-grained keywords as a substring and would otherwise silently
    resolve to "unknown" (a real bug this comment exists because of: it
    dropped every non-valve detection from the coarse model in initial
    testing)."""
    lowered = class_name.lower()
    if lowered in _COARSE_CLASS_NAMES:
        return lowered
    for keyword, component_type in _KEYWORD_TO_TYPE.items():
        if keyword in lowered:
            return component_type
    return "unknown"


def weights_available(weights_path: Path | None = None) -> bool:
    """weights_path=None resolves DEFAULT_WEIGHTS_PATH at call time (module
    global lookup), not as a bound default argument -- Python binds default
    argument values once at function-definition time, so `weights_path: Path =
    DEFAULT_WEIGHTS_PATH` would silently keep using whatever
    DEFAULT_WEIGHTS_PATH was when this module first loaded, ignoring any later
    reassignment (the exact bug pipeline.py's caller-side fix works around --
    fixed here too so this footgun doesn't resurface for some other caller)."""
    return (weights_path or DEFAULT_WEIGHTS_PATH).exists()


def _iter_tile_offsets(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    stride = tile_size - overlap
    xs = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
    if xs[-1] + tile_size < width:
        xs.append(width - tile_size)
    if ys[-1] + tile_size < height:
        ys.append(height - tile_size)
    return [(x, y) for y in ys for x in xs]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _merge_overlapping_detections(
    detections: list[dict], iou_threshold: float = MERGE_IOU_THRESHOLD
) -> list[dict]:
    """Same symbol detected in more than one overlapping tile collapses to the
    single highest-confidence box, regardless of predicted class — a detector
    disagreeing with itself across tile boundaries on class but not location
    is still one physical symbol, not two."""
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict] = []
    for det in ordered:
        if any(_iou(det["xyxy"], other["xyxy"]) > iou_threshold for other in kept):
            continue
        kept.append(det)
    return kept


def _run_model(model, image: np.ndarray, confidence: float, offset: tuple[int, int] = (0, 0)) -> list[dict]:
    ox, oy = offset
    detections = []
    for result in model.predict(image, conf=confidence, verbose=False):
        names = result.names
        for box in result.boxes:
            x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "xyxy": (x0 + ox, y0 + oy, x1 + ox, y1 + oy),
                    "class_name": names[int(box.cls[0].item())],
                    "confidence": float(box.conf[0].item()),
                }
            )
    return detections


def detect_symbols(
    image: np.ndarray,
    weights_path: Path | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[DetectedShape]:
    """Run the trained YOLO model on a rendered page image (as produced by
    pdf_to_image.pdf_to_images). Returns DetectedShape objects with `kind` set
    to a lowercase, space-stripped version of the dataset class name (e.g.
    "gate_valve") — component_type is resolved separately via
    class_name_to_component_type, not derived from `kind` the way raster/vector
    shapes elsewhere in this package derive it from a fixed small vocabulary.

    weights_path=None resolves DEFAULT_WEIGHTS_PATH at call time rather than as
    a bound default argument -- see weights_available()'s docstring for why
    that distinction matters; this is the same fix applied here too.

    Images larger than one tile are sliced into overlapping TILE_SIZE crops
    (see module docstring for why: the model was trained on pre-cropped
    640x640 tiles, and small P&ID symbols disappear if the whole page is
    resized down to fit that size in a single pass) — detections are merged
    back into full-image coordinates and deduplicated across tile overlaps."""
    resolved_weights_path = weights_path or DEFAULT_WEIGHTS_PATH
    if not resolved_weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {resolved_weights_path} — train the model first "
            f"(see README 'Closing the Valve Gap Further') or pass an explicit path."
        )

    from ultralytics import YOLO  # deferred: heavy import, only needed when this path is used

    model = YOLO(str(resolved_weights_path))
    height, width = image.shape[:2]

    if height <= TILE_SIZE and width <= TILE_SIZE:
        detections = _run_model(model, image, confidence)
    else:
        detections = []
        for x, y in _iter_tile_offsets(width, height, TILE_SIZE, TILE_OVERLAP):
            crop = image[y : y + TILE_SIZE, x : x + TILE_SIZE]
            detections.extend(_run_model(model, crop, confidence, offset=(x, y)))
        detections = _merge_overlapping_detections(detections)

    shapes = []
    for det in detections:
        x0, y0, x1, y1 = det["xyxy"]
        shapes.append(
            DetectedShape(
                shape_id=len(shapes),
                kind=det["class_name"].lower().replace(" ", "_"),
                bbox=(x0, y0, x1 - x0, y1 - y0),
                center=((x0 + x1) // 2, (y0 + y1) // 2),
                contour=np.empty((0, 1, 2), dtype=np.int32),
            )
        )
    return shapes
