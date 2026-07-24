import json
from unittest.mock import patch

import networkx as nx
import pytest

from evaluation.score_cv_accuracy import _normalize_tag, score


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_normalize_tag_happy_path():
    assert _normalize_tag("pi 715a") == "PI 715A"
    assert _normalize_tag("  FT-101  ") == "FT-101"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_normalize_tag_collapses_internal_whitespace_edge_case():
    assert _normalize_tag("PI   715A") == "PI 715A"


def _fake_graph():
    graph = nx.Graph()
    graph.add_node("PI 715A", tag="PI 715A", component_type="pressure_indicator")
    graph.add_node("UNLABELED-0", tag=None, component_type="instrument")
    graph.add_node("MV-999-99", tag="MV-999-99", component_type="valve")  # not in ground truth
    return graph


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_score_computes_precision_recall_against_ground_truth(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "components": [
                    {"tag": "PI 715A", "category": "instrument", "symbol": "circle_bubble"},
                    {"tag": "PSV 715A", "category": "instrument", "symbol": "circle_bubble"},
                    {"tag": "MV-715-01", "category": "valve", "symbol": "bowtie"},
                ]
            }
        )
    )

    with patch("evaluation.score_cv_accuracy.extract_pid_graph", return_value=_fake_graph()):
        result = score(tmp_path / "fake.pdf", 0, gt_path, llm_ocr_assist=False)

    ta = result["tag_accuracy"]
    assert ta["true_positives"] == ["PI 715A"]
    assert ta["false_positives"] == ["MV-999-99"]
    assert set(ta["false_negatives_sample"]) == {"PSV 715A", "MV-715-01"}
    assert ta["precision"] == round(1 / 2, 3)
    assert ta["recall"] == round(1 / 3, 3)


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_score_no_resolved_tags_edge_case(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({"components": [{"tag": "PI 715A", "category": "instrument", "symbol": "circle_bubble"}]}))

    empty_graph = nx.Graph()
    empty_graph.add_node("UNLABELED-0", tag=None, component_type="instrument")

    with patch("evaluation.score_cv_accuracy.extract_pid_graph", return_value=empty_graph):
        result = score(tmp_path / "fake.pdf", 0, gt_path, llm_ocr_assist=False)

    ta = result["tag_accuracy"]
    assert ta["precision"] is None  # no tags reported at all — division by zero avoided, not zero
    assert ta["recall"] == 0.0
