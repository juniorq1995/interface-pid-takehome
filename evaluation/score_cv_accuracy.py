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

No bounding-box ground truth exists (this was hand-annotated by eye from a
rendered image, not pixel-labeled), so this can't score localization (IoU) —
only "was this exact tag found somewhere." That's a real, disclosed limitation
of this evaluation, not hidden.

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


def score(pdf_path: Path, page: int, ground_truth_path: Path, llm_ocr_assist: bool) -> dict:
    ground_truth = json.loads(ground_truth_path.read_text())
    gt_components = ground_truth["components"]
    gt_by_category = Counter(c["category"] for c in gt_components)
    gt_tags = {_normalize_tag(c["tag"]) for c in gt_components}

    graph = extract_pid_graph(pdf_path, page=page, dpi=300, llm_ocr_assist=llm_ocr_assist)

    detected_by_category = Counter(d.get("component_type", "unknown") for _, d in graph.nodes(data=True))
    # Ground truth categories are coarse (instrument/valve/equipment); collapse
    # detected component_type (e.g. "pressure_indicator") the same way for a fair count.
    coarse_map = {"instrument": "instrument", "valve": "valve", "safety_valve": "valve", "tank": "equipment"}
    from src.component_types import coarse_category

    detected_coarse = Counter()
    for _, d in graph.nodes(data=True):
        coarse = coarse_map.get(d.get("component_type"), coarse_category(d.get("component_type")) or "unknown")
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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = score(args.pdf, args.page, args.ground_truth, args.llm_ocr_assist)

    print(f"CV accuracy score — {args.pdf} page {args.page} (llm_ocr_assist={args.llm_ocr_assist})")
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
