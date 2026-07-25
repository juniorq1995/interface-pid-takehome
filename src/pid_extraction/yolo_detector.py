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

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "datasets" / "runs" / "pid_valve_v1" / "weights" / "best.pt"
DEFAULT_CONFIDENCE = 0.25

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


def class_name_to_component_type(class_name: str) -> str:
    lowered = class_name.lower()
    for keyword, component_type in _KEYWORD_TO_TYPE.items():
        if keyword in lowered:
            return component_type
    return "unknown"


def weights_available(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> bool:
    return weights_path.exists()


def detect_symbols(
    image: np.ndarray,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[DetectedShape]:
    """Run the trained YOLO model on a rendered page image (as produced by
    pdf_to_image.pdf_to_images). Returns DetectedShape objects with `kind` set
    to a lowercase, space-stripped version of the dataset class name (e.g.
    "gate_valve") — component_type is resolved separately via
    class_name_to_component_type, not derived from `kind` the way raster/vector
    shapes elsewhere in this package derive it from a fixed small vocabulary."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_path} — train the model first "
            f"(see README 'Closing the Valve Gap Further') or pass an explicit path."
        )

    from ultralytics import YOLO  # deferred: heavy import, only needed when this path is used

    model = YOLO(str(weights_path))
    results = model.predict(image, conf=confidence, verbose=False)

    shapes = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            class_name = names[class_id]
            shapes.append(
                DetectedShape(
                    shape_id=len(shapes),
                    kind=class_name.lower().replace(" ", "_"),
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    center=((x0 + x1) // 2, (y0 + y1) // 2),
                    contour=np.empty((0, 1, 2), dtype=np.int32),
                )
            )
    return shapes
