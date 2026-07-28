"""Real, non-circular bbox localization scoring against a hand-verified sample.

score_cv_accuracy.py already covers category coverage (counts) and tag
accuracy (exact text). Neither measures whether a detected shape is actually
*where* the real symbol is -- this closes that gap using
evaluation/ground_truth_bbox_sample.json, a 12-component sample located by
hand from coordinate-gridded crops of the real rendered pages (see that
file's "_methodology" field for exactly how, and why it isn't circular).

Matching is by nearest-center-distance within the same coarse category, not
by tag -- tag resolution is currently a separate, already-documented 0%
gap (see README), so almost no detected node carries a tag to match on.
This script answers a narrower, still-real question: when the pipeline
detects *something* in the right category near a known real symbol, how
tightly does that detection's box overlap the symbol's actual extent?

Usage:
    python evaluation/score_bbox_precision.py [--max-center-distance-px 120]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.component_types import project_coarse_category
from src.pid_extraction.pipeline import extract_pid_graph


def _coarse(component_type: str | None) -> str:
    return project_coarse_category(component_type) or "unknown"

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth_bbox_sample.json"
PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "pid" / "diagram.pdf"


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, aw, ah = a
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = b
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def score(ground_truth_path: Path, max_center_distance_px: float) -> dict[str, Any]:
    gt = json.loads(ground_truth_path.read_text())
    by_page: dict[int, list[dict[str, Any]]] = {}
    for c in gt["components"]:
        by_page.setdefault(c["page"], []).append(c)

    results = []
    for page, components in sorted(by_page.items()):
        graph = extract_pid_graph(PDF_PATH, page=page, dpi=300, use_yolo=True)
        detections_by_category: dict[str, list[list[float]]] = {}
        for _, d in graph.nodes(data=True):
            detections_by_category.setdefault(_coarse(d.get("component_type")), []).append(d["bbox"])

        for c in components:
            gt_bbox = c["bbox"]
            gt_center = _center(gt_bbox)
            candidates = detections_by_category.get(c["category"], [])
            best = None
            best_dist = float("inf")
            for det_bbox in candidates:
                d = _dist(gt_center, _center(det_bbox))
                if d < best_dist:
                    best_dist, best = d, det_bbox
            matched = best is not None and best_dist <= max_center_distance_px
            iou = _iou(gt_bbox, best) if matched else 0.0
            results.append(
                {
                    "page": page,
                    "category": c["category"],
                    "tag": c["tag"],
                    "gt_bbox": gt_bbox,
                    "matched": matched,
                    "nearest_center_distance_px": round(best_dist, 1) if best is not None else None,
                    "matched_detection_bbox": best if matched else None,
                    "iou": round(iou, 3),
                }
            )

    matched_results = [r for r in results if r["matched"]]
    return {
        "sample_size": len(results),
        "matched_count": len(matched_results),
        "unmatched_count": len(results) - len(matched_results),
        "mean_iou_over_matched": round(sum(r["iou"] for r in matched_results) / len(matched_results), 3)
        if matched_results
        else None,
        "mean_iou_over_all_including_unmatched_as_zero": round(sum(r["iou"] for r in results) / len(results), 3)
        if results
        else None,
        "per_component": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--max-center-distance-px", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = score(args.ground_truth, args.max_center_distance_px)

    print(f"Bbox localization score -- sample of {result['sample_size']} hand-verified components")
    print()
    print(f"Matched (nearest same-category detection within {args.max_center_distance_px}px): "
          f"{result['matched_count']}/{result['sample_size']}")
    print(f"Mean IoU over matched components: {result['mean_iou_over_matched']}")
    print(f"Mean IoU over all components (unmatched counted as 0): "
          f"{result['mean_iou_over_all_including_unmatched_as_zero']}")
    print()
    for r in result["per_component"]:
        status = f"IoU={r['iou']}" if r["matched"] else f"UNMATCHED (nearest {r['category']} was {r['nearest_center_distance_px']}px away)"
        print(f"  page{r['page']} {r['category']:10s} {r['tag']:14s} {status}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
