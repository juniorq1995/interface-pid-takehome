import numpy as np
import pytest

from src.component_types import SHAPE_TO_TYPE
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


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_build_graph_component_type_override_used_for_unlabeled_shape():
    """A YOLO-detected shape has no OCR tag and a `kind` ("gate_valve") that
    isn't in SHAPE_TO_TYPE's small heuristic vocabulary — without the override
    it would silently resolve to "unknown", discarding the trained model's
    class prediction."""
    shapes = [_shape(0, kind="gate_valve")]
    tags = {0: (None, "")}
    graph = build_graph(shapes, tags, edges=[], component_type_overrides={0: "valve"})

    assert graph.nodes["UNLABELED-0"]["component_type"] == "valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_build_graph_tag_derived_type_refines_within_same_category_edge_case():
    """A tag-derived type that agrees with the shape's own coarse category
    (here: both "valve") is a legitimate refinement -- generic "valve" from
    the YOLO override sharpens to the specific "pressure_control_valve" the
    tag names. This is the case the original (pre-fix) version of this test
    was checking, just with a same-category tag instead of a cross-category
    one -- see the contradiction test below for why that distinction now
    matters."""
    shapes = [_shape(0, kind="gate_valve")]
    tags = {0: ("PCV-101", "PCV-101")}
    graph = build_graph(shapes, tags, edges=[], component_type_overrides={0: "valve"})

    assert graph.nodes["PCV-101"]["component_type"] == "pressure_control_valve"


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_build_graph_contradictory_tag_type_does_not_override_geometry_regression():
    """Real bug found and fixed live this session: a resolved tag used to
    unconditionally win over an already-known geometric/YOLO type -- the
    reasoning being "printed tag text is ground truth." That broke on real
    data: llm_ocr_assist's oversized-crop framing problem caused a vessel
    shape (component_type "tank" via SHAPE_TO_TYPE) to get misread as a
    nearby flow transmitter's tag ("FT-101"), and the old priority order let
    that wrong-but-real tag silently reclassify the vessel from "tank" to
    "flow_transmitter" -- measured: page 0 equipment coverage 2/2 -> 0/2.
    Fixed: a tag-derived type is only trusted when its coarse category
    (valve/instrument/equipment) agrees with the geometry-derived type's --
    a genuine cross-category contradiction falls back to geometry instead,
    since this document's geometry-only typing is independently measured at
    100% category coverage (see README) while tag-reading is not."""
    shapes = [_shape(0, kind="rectangle")]  # SHAPE_TO_TYPE["rectangle"] == "tank"
    tags = {0: ("FT-101", "FT-101")}  # real prefix, but a different coarse category

    graph = build_graph(shapes, tags, edges=[])

    node_data = graph.nodes["FT-101"]
    assert node_data["component_type"] == "tank"  # geometry wins, not "flow_transmitter"
    assert node_data["tag"] == "FT-101"  # the tag text itself is still recorded honestly


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_build_graph_missing_override_falls_back_to_shape_to_type_failure_path():
    shapes = [_shape(0, kind="circle")]
    tags = {0: (None, "")}
    graph = build_graph(shapes, tags, edges=[], component_type_overrides={})

    assert graph.nodes["UNLABELED-0"]["component_type"] == SHAPE_TO_TYPE["circle"]


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_build_graph_tag_type_used_when_geometry_has_no_opinion_happy_path():
    """When geometry gives no confident type at all (kind not in
    SHAPE_TO_TYPE, no override -- exactly Phase 1's "unknown" target case),
    a resolved tag has nothing to contradict and is used as-is."""
    shapes = [_shape(0, kind="unknown")]
    tags = {0: ("TI-101", "TI-101")}
    graph = build_graph(shapes, tags, edges=[])

    assert graph.nodes["TI-101"]["component_type"] == "temperature_indicator"
