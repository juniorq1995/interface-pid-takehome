"""Score the P&ID component-detection pipeline against a hand-annotated ground
truth, so CV accuracy is a tracked number instead of a vibe.

Two separate metrics, deliberately not blended into one score:

1. Category coverage — for each component category (instrument/valve/equipment),
   detected count vs ground-truth count. This measures "did we find roughly the
   right number of symbols," independent of whether we read their tags — useful
   because tag OCR and symbol detection are genuinely separate failure modes
   (see README "Component-graph extraction").
2. Tag accuracy — precision/recall/F1 on exact (normalized) tag text, only over
   components the pipeline actually resolved a tag for. Measures OCR/LLM-assist
   quality specifically, independent of symbol-detection coverage.

This ground truth is text/category only (hand-annotated by eye from a
rendered image, not pixel-labeled), so it can't score localization (IoU) —
only "was this exact tag found somewhere." A separate, smaller hand-verified
bbox sample now exists for that (see score_bbox_precision.py and README
"Real bbox localization precision") — deliberately kept as a second, narrower
script rather than merged in here, since it answers a different question
(overlap quality on a 12-component sample) with a different matching strategy
(nearest-detection-by-category, not exact tag) than this script's exhaustive
tag-based scoring.

Usage:
    python evaluation/score_cv_accuracy.py [--pdf PATH] [--page N] [--llm-ocr-assist]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pid_extraction.pipeline import extract_pid_graph

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth_page0.json"


def _normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip().upper())


def score(
    pdf_path: Path, page: int, ground_truth_path: Path, llm_ocr_assist: bool, use_yolo: bool = False,
    yolo_weights_path: Path | None = None, symbol_type_assist: bool = False,
) -> dict:
    ground_truth = json.loads(ground_truth_path.read_text())
    gt_components = ground_truth["components"]
    gt_by_category = Counter(c["category"] for c in gt_components)
    gt_tags = {_normalize_tag(c["tag"]) for c in gt_components}

    graph = extract_pid_graph(
        pdf_path, page=page, dpi=300, llm_ocr_assist=llm_ocr_assist, use_yolo=use_yolo,
        yolo_weights_path=yolo_weights_path, symbol_type_assist=symbol_type_assist,
    )

    detected_by_category = Counter(d.get("component_type", "unknown") for _, d in graph.nodes(data=True))
    # Ground truth categories are coarse (instrument/valve/equipment); collapse
    # detected component_type (e.g. "pressure_indicator") the same way for a fair count.
    # Real bug found and fixed here: this used to be a small 4-entry local dict
    # that didn't cover most of TAG_PREFIX_TYPES' ~44 fine-grained values, so a
    # correctly tag-resolved "pressure_indicator"/"pressure_differential_indicator"
    # (etc.) silently vanished from the "instrument" row entirely once
    # llm_ocr_assist started actually resolving real tags -- measured: page 0
    # instrument coverage showed 2/6 despite all 6 real instruments having been
    # correctly read. Now uses the shared, complete project_coarse_category().
    from src.component_types import project_coarse_category

    detected_coarse = Counter()
    for _, d in graph.nodes(data=True):
        coarse = project_coarse_category(d.get("component_type")) or "unknown"
        detected_coarse[coarse] += 1

    category_coverage = {}
    for category, gt_count in gt_by_category.items():
        found = detected_coarse.get(category, 0)
        category_coverage[category] = {
            "ground_truth_count": gt_count,
            "detected_count": found,
            "coverage_pct": round(100 * min(found, gt_count) / gt_count, 1) if gt_count else None,
        }

    resolved_tags = {_normalize_tag(d["tag"]) for _, d in graph.nodes(data=True) if d.get("tag")}
    true_positives = resolved_tags & gt_tags
    false_positives = resolved_tags - gt_tags
    false_negatives = gt_tags - resolved_tags

    precision = len(true_positives) / len(resolved_tags) if resolved_tags else None
    recall = len(true_positives) / len(gt_tags) if gt_tags else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None

    return {
        "pdf": str(pdf_path),
        "page": page,
        "llm_ocr_assist": llm_ocr_assist,
        "category_coverage": category_coverage,
        "tag_accuracy": {
            "resolved_tags_reported": len(resolved_tags),
            "ground_truth_tags": len(gt_tags),
            "true_positives": sorted(true_positives),
            "false_positives": sorted(false_positives),
            "false_negatives_sample": sorted(false_negatives)[:10],
            "false_negatives_count": len(false_negatives),
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("data/pid/diagram.pdf"))
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--llm-ocr-assist", action="store_true")
    parser.add_argument("--use-yolo", action="store_true", help="Add the trained YOLO detector as an ensemble source")
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Override YOLO checkpoint path (default: yolo_detector.DEFAULT_WEIGHTS_PATH)")
    parser.add_argument("--symbol-type-assist", action="store_true", help="Phase 1: classify shapes the geometric heuristics couldn't type, via local vision model")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = score(
        args.pdf, args.page, args.ground_truth, args.llm_ocr_assist, args.use_yolo, args.yolo_weights,
        args.symbol_type_assist,
    )

    print(
        f"CV accuracy score — {args.pdf} page {args.page} "
        f"(llm_ocr_assist={args.llm_ocr_assist}, use_yolo={args.use_yolo}, symbol_type_assist={args.symbol_type_assist})"
    )
    print()
    print("Category coverage (symbol found, regardless of tag correctness):")
    for category, stats in result["category_coverage"].items():
        print(f"  {category:12s} {stats['detected_count']:3d}/{stats['ground_truth_count']:3d}  ({stats['coverage_pct']}%)")
    print()
    ta = result["tag_accuracy"]
    print("Tag accuracy (exact normalized text match against ground truth):")
    print(f"  precision: {ta['precision']}   recall: {ta['recall']}   f1: {ta['f1']}")
    print(f"  true positives:  {ta['true_positives']}")
    print(f"  false positives: {ta['false_positives']}")
    print(f"  false negatives: {ta['false_negatives_count']} (sample: {ta['false_negatives_sample']})")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
