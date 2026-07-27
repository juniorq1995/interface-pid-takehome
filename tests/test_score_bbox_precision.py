import json
from unittest.mock import patch

import networkx as nx
import pytest

from evaluation.score_bbox_precision import _coarse, _dist, _iou, score


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_iou_identical_boxes_is_one():
    box = [10, 10, 50, 50]
    assert _iou(box, box) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_iou_disjoint_boxes_is_zero_edge_case():
    assert _iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_iou_partial_overlap_happy_path():
    # two 10x10 boxes overlapping in a 5x10 strip: intersection=50, union=150
    result = _iou([0, 0, 10, 10], [5, 0, 10, 10])
    assert result == pytest.approx(50 / 150)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_dist_pythagorean_happy_path():
    assert _dist((0, 0), (3, 4)) == pytest.approx(5.0)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_coarse_maps_tank_to_equipment_and_passes_through_known_categories():
    assert _coarse("tank") == "equipment"
    assert _coarse("safety_valve") == "valve"
    assert _coarse("instrument") == "instrument"


def _fake_graph_page0():
    graph = nx.Graph()
    # exact match for the instrument ground-truth entry below
    graph.add_node("UNLABELED-0", component_type="instrument", bbox=[100, 100, 50, 50])
    # a tank-typed node far away -- should map to "equipment" category but not match (too far)
    graph.add_node("UNLABELED-1", component_type="tank", bbox=[5000, 5000, 100, 100])
    return graph


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_score_matches_nearest_same_coarse_category_detection(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "components": [
                    {"page": 0, "category": "instrument", "tag": "PI 715A", "bbox": [100, 100, 50, 50]},
                ]
            }
        )
    )

    with patch("evaluation.score_bbox_precision.extract_pid_graph", return_value=_fake_graph_page0()):
        result = score(gt_path, max_center_distance_px=120.0)

    assert result["matched_count"] == 1
    assert result["per_component"][0]["iou"] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_score_no_candidate_in_category_is_unmatched_not_a_crash_edge_case(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "components": [
                    {"page": 0, "category": "valve", "tag": "MV-1", "bbox": [0, 0, 10, 10]},
                ]
            }
        )
    )

    with patch("evaluation.score_bbox_precision.extract_pid_graph", return_value=_fake_graph_page0()):
        result = score(gt_path, max_center_distance_px=120.0)

    assert result["matched_count"] == 0
    assert result["per_component"][0]["matched"] is False
    assert result["per_component"][0]["iou"] == 0.0
    assert result["mean_iou_over_matched"] is None


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_score_nearest_detection_beyond_max_distance_is_unmatched_edge_case(tmp_path):
    """A same-category detection exists but is too far away -- must not be
    silently accepted as a match just because it's the nearest one."""
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "components": [
                    {"page": 0, "category": "equipment", "tag": "F-999", "bbox": [0, 0, 10, 10]},
                ]
            }
        )
    )

    with patch("evaluation.score_bbox_precision.extract_pid_graph", return_value=_fake_graph_page0()):
        result = score(gt_path, max_center_distance_px=120.0)

    assert result["matched_count"] == 0
    assert result["per_component"][0]["nearest_center_distance_px"] > 120.0
