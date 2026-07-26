"""Build a combined, coarse-class YOLO dataset from Digitize-PID + Roboflow
P&ID Symbols R2, targeting exactly the 3 categories this project's own
ground truth and component_types.py actually distinguish: valve / instrument
/ equipment. Neither source dataset's own fine-grained taxonomy (32 or 180
classes) matches what this project needs — merging both into one coarse
label space gives the detector far more training signal per class than
either source alone (Digitize-PID's valve-bowtie classes alone sum to
~15,500 instances across 400 full-page synthetic P&IDs, vs a few hundred
per class fragmented across Roboflow's 180 categories).

Digitize-PID's class-to-category mapping below was built by direct visual
inspection of one real crop per class ID (see the contact sheet generated
during this session) — the dataset ships with numeric-only labels and no
public class-name mapping (the source paper's PDF isn't machine-readable,
and the HuggingFace card doesn't list one either). Ambiguous classes (plain
triangles, the very common "double line" class, flow-direction arrows) are
deliberately excluded rather than guessed into a bucket.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIGITIZE_ROOT = REPO_ROOT / "datasets" / "digitize-pid-yolo" / "DigitizePID_Dataset"
ROBOFLOW_ROOT = REPO_ROOT / "datasets" / "pid-symbols-r2"
OUTPUT_ROOT = REPO_ROOT / "datasets" / "merged-coarse"

COARSE_CLASSES = ["valve", "instrument", "equipment"]
VALVE, INSTRUMENT, EQUIPMENT = 0, 1, 2

# Visual mapping — see module docstring. Classes not listed here are
# excluded (ambiguous/ill-fitting, not guessed into a bucket).
DIGITIZE_PID_CLASS_MAP = {
    0: VALVE, 1: VALVE, 2: VALVE, 4: VALVE, 5: VALVE, 6: VALVE,
    8: VALVE, 10: VALVE, 11: VALVE, 12: VALVE, 13: VALVE,
    14: INSTRUMENT, 15: INSTRUMENT, 16: INSTRUMENT, 17: INSTRUMENT,
    18: INSTRUMENT, 24: INSTRUMENT, 25: INSTRUMENT, 26: INSTRUMENT,
    27: INSTRUMENT, 28: INSTRUMENT, 29: INSTRUMENT, 30: INSTRUMENT,
    31: INSTRUMENT,
    19: EQUIPMENT, 21: EQUIPMENT, 22: EQUIPMENT,
    # excluded: 3, 7, 9 (ambiguous triangles), 20 (very common, unclear
    # meaning), 23 (flow-direction arrow, not a component)
}


def _roboflow_class_map(names: list[str]) -> dict[int, int]:
    from src.pid_extraction.yolo_detector import class_name_to_component_type

    coarse_by_type = {
        "valve": VALVE, "safety_valve": VALVE, "relief_valve": VALVE,
        "instrument": INSTRUMENT,
        "tank": EQUIPMENT, "pump": EQUIPMENT, "compressor": EQUIPMENT,
        "turbine": EQUIPMENT, "heat_exchanger": EQUIPMENT,
    }
    mapping = {}
    for class_id, name in enumerate(names):
        component_type = class_name_to_component_type(name)
        if component_type in coarse_by_type:
            mapping[class_id] = coarse_by_type[component_type]
    return mapping


def _relabel_file(src_label: Path, dst_label: Path, class_map: dict[int, int]) -> int:
    kept_lines = []
    for line in src_label.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        original_class = int(parts[0])
        if original_class not in class_map:
            continue
        kept_lines.append(" ".join([str(class_map[original_class]), *parts[1:]]))
    dst_label.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return len(kept_lines)


def _copy_split(
    *,
    image_files: list[Path],
    label_dir: Path,
    class_map: dict[int, int],
    out_split: str,
    prefix: str,
) -> tuple[int, int]:
    images_out = OUTPUT_ROOT / out_split / "images"
    labels_out = OUTPUT_ROOT / out_split / "labels"
    kept_images = 0
    kept_boxes = 0
    for image_path in image_files:
        label_path = label_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            continue
        dst_name = f"{prefix}_{image_path.stem}"
        dst_label = labels_out / f"{dst_name}.txt"
        box_count = _relabel_file(label_path, dst_label, class_map)
        if box_count == 0:
            dst_label.unlink(missing_ok=True)
            continue
        shutil.copy(image_path, images_out / f"{dst_name}{image_path.suffix}")
        kept_images += 1
        kept_boxes += box_count
    return kept_images, kept_boxes


def build() -> dict:
    for split in ("train", "valid"):
        (OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {}

    # Digitize-PID: its own train/val split, 4:1 ratio, reused as-is.
    for src_split, out_split in (("train", "train"), ("val", "valid")):
        images = sorted((DIGITIZE_ROOT / "images" / src_split).glob("*.jpg"))
        kept_images, kept_boxes = _copy_split(
            image_files=images,
            label_dir=DIGITIZE_ROOT / "labels" / src_split,
            class_map=DIGITIZE_PID_CLASS_MAP,
            out_split=out_split,
            prefix="digitize",
        )
        stats[f"digitize_{out_split}"] = {"images": kept_images, "boxes": kept_boxes}

    # Roboflow P&ID Symbols R2: read its own class names from data_abs.yaml.
    import yaml

    data_yaml = yaml.safe_load((ROBOFLOW_ROOT / "data_abs.yaml").read_text())
    roboflow_map = _roboflow_class_map(data_yaml["names"])
    for src_split, out_split in (("train", "train"), ("valid", "valid")):
        images = sorted((ROBOFLOW_ROOT / src_split / "images").glob("*"))
        kept_images, kept_boxes = _copy_split(
            image_files=images,
            label_dir=ROBOFLOW_ROOT / src_split / "labels",
            class_map=roboflow_map,
            out_split=out_split,
            prefix="roboflow",
        )
        stats[f"roboflow_{out_split}"] = {"images": kept_images, "boxes": kept_boxes}

    data_yaml_out = OUTPUT_ROOT / "data.yaml"
    data_yaml_out.write_text(
        f"path: {OUTPUT_ROOT}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"\n"
        f"nc: {len(COARSE_CLASSES)}\n"
        f"names: {COARSE_CLASSES}\n"
    )
    return stats


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    result = build()
    print(json.dumps(result, indent=2))
