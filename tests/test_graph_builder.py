import numpy as np
import pytest

from src.pid_extraction.graph_builder import build_graph
from src.pid_extraction.shape_detection import DetectedShape


def _shape(shape_id, kind="circle"):
    return DetectedShape(shape_id=shape_id, kind=kind, bbox=(0, 0, 10, 10), center=(5, 5), contour=np.empty((0, 1, 2), dtype=np.int32))


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_build_graph_happy_path_resolved_tag():
    shapes = [_shape(0)]
    tags = {0: ("PI-101", "PI-101")}
    graph = build_graph(shapes, tags, edges=[])

    assert list(graph.nodes) == ["PI-101"]
    assert graph.nodes["PI-101"]["tag"] == "PI-101"
    assert graph.nodes["PI-101"]["tag_confidence"] == "ocr_matched"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_build_graph_unresolved_shapes_get_unique_labels_edge_case():
    shapes = [_shape(0), _shape(1)]
    tags = {0: (None, ""), 1: (None, "")}
    graph = build_graph(shapes, tags, edges=[])

    assert sorted(graph.nodes) == ["UNLABELED-0", "UNLABELED-1"]


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_build_graph_duplicate_tags_get_distinct_nodes():
    """Regression: two distinct shapes resolving to the same tag used to collide
    on the same nx.Graph node label — the second node_add call silently
    overwrote the first, losing a real physical component. Fixed by
    disambiguating the label for the 2nd+ occurrence while keeping the `tag`
    attribute identical on both, so cross_reference.py can still find and flag
    the collision (DUPLICATE_TAG) instead of it vanishing silently."""
    shapes = [_shape(0), _shape(1), _shape(2)]
    tags = {0: ("P-101", "P-101"), 1: ("P-101", "P-101"), 2: ("T-1", "T-1")}
    graph = build_graph(shapes, tags, edges=[])

    assert graph.number_of_nodes() == 3  # not collapsed to 2
    labels = sorted(graph.nodes)
    assert "P-101" in labels
    assert any(label.startswith("P-101 (dup") for label in labels)
    # Both P-101 nodes keep the same real tag attribute — only the graph key differs.
    p101_nodes = [d for _, d in graph.nodes(data=True) if d["tag"] == "P-101"]
    assert len(p101_nodes) == 2


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_build_graph_edges_attach_to_correct_shape_after_disambiguation():
    # Edge lookup is by shape_id, not label — confirm disambiguated labels
    # still get the right edges, not swapped or dropped.
    shapes = [_shape(0), _shape(1), _shape(2)]
    tags = {0: ("P-101", "P-101"), 1: ("P-101", "P-101"), 2: ("T-1", "T-1")}
    graph = build_graph(shapes, tags, edges=[(1, 2)])  # second P-101 connects to T-1

    dup_label = next(label for label in graph.nodes if label.startswith("P-101 (dup"))
    assert graph.has_edge(dup_label, "T-1")
    assert not graph.has_edge("P-101", "T-1")
