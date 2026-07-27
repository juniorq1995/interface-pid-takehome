from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.pid_extraction import yolo_detector
from src.pid_extraction.pipeline import _extract_page_graph, _merge_shapes
from src.pid_extraction.shape_detection import DetectedShape


def _shape(kind):
    return DetectedShape(shape_id=0, kind=kind, bbox=(0, 0, 10, 10), center=(5, 5), contour=np.empty((0, 1, 2), dtype=np.int32))


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_merge_shapes_no_yolo_shapes_returns_empty_overrides():
    shapes, overrides = _merge_shapes([_shape("circle")], [_shape("bowtie")])
    assert len(shapes) == 2
    assert overrides == {}


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_merge_shapes_yolo_shapes_get_component_type_overrides():
    """YOLO shapes are appended last and re-numbered into the shared shape_id
    space; each gets an override entry keyed by its *new* id, not shape_id=0
    that every input DetectedShape carries before merging."""
    raster = [_shape("circle")]
    vector = [_shape("bowtie")]
    yolo = [_shape("gate_valve"), _shape("pressure_transmitter")]

    shapes, overrides = _merge_shapes(raster, vector, yolo_shapes=yolo)

    assert len(shapes) == 4
    assert [s.shape_id for s in shapes] == [0, 1, 2, 3]
    # vector shapes are ordered before raster shapes, then yolo shapes last
    assert shapes[0].kind == "bowtie"
    assert shapes[1].kind == "circle"
    assert shapes[2].kind == "gate_valve"
    assert shapes[3].kind == "pressure_transmitter"
    assert overrides == {2: "valve", 3: "instrument"}


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_merge_shapes_yolo_only_unmapped_class_falls_back_to_unknown_edge_case():
    shapes, overrides = _merge_shapes([], yolo_shapes=[_shape("manhole")])
    assert overrides == {0: "unknown"}


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_page_graph_picks_up_runtime_default_weights_path_change(monkeypatch, tmp_path):
    """Regression: Python binds a function's default argument values once at
    definition time. Reassigning yolo_detector.DEFAULT_WEIGHTS_PATH at runtime
    used to be silently ignored by any caller that didn't pass weights_path
    explicitly (including pipeline.py's own yolo call) — every "checkpoint
    comparison" done this way kept re-testing whatever model was default when
    the module was first imported, regardless of later reassignment. Confirms
    the fix: _extract_page_graph now reads the module attribute live."""
    new_default = tmp_path / "new_default.pt"
    monkeypatch.setattr(yolo_detector, "DEFAULT_WEIGHTS_PATH", new_default)

    with (
        patch("src.pid_extraction.pipeline.weights_available", return_value=True) as mock_available,
        patch("src.pid_extraction.pipeline.detect_yolo_symbols", return_value=[]) as mock_detect,
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
    ):
        _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), use_yolo=True)

    mock_available.assert_called_once_with(new_default)
    mock_detect.assert_called_once()
    assert mock_detect.call_args[0][1] == new_default


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_extract_page_graph_explicit_weights_path_overrides_default():
    explicit_path = Path("/explicit/weights.pt")

    with (
        patch("src.pid_extraction.pipeline.weights_available", return_value=True) as mock_available,
        patch("src.pid_extraction.pipeline.detect_yolo_symbols", return_value=[]) as mock_detect,
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
    ):
        _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), use_yolo=True, yolo_weights_path=explicit_path)

    mock_available.assert_called_once_with(explicit_path)
    assert mock_detect.call_args[0][1] == explicit_path


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_fills_component_type_for_unknown_shape():
    """Phase 1 wiring: a shape with no tag-resolved type, no YOLO override,
    and a kind outside SHAPE_TO_TYPE's vocabulary (kind="unknown", so it would
    otherwise reach the graph as component_type "unknown") gets classified by
    the local vision model instead when symbol_type_assist=True."""
    shape = _shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm", return_value=("valve", "gate_valve", "valve, gate_valve")) as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), symbol_type_assist=True)

    mock_classify.assert_called_once()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "gate_valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_never_overrides_tag_resolved_type_edge_case():
    """A shape with a real, tag-resolved type must never be sent to the vision
    model at all -- symbol_type_assist only fills gaps, it doesn't second-guess
    a cheaper/more-certain path."""
    shape = _shape("circle")  # circle already maps to "instrument" via SHAPE_TO_TYPE

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=("PI-101", "PI-101")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm") as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), symbol_type_assist=True)

    mock_classify.assert_not_called()
    node_data = graph.nodes["PI-101"]
    assert node_data["component_type"] == "pressure_indicator"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_off_by_default_edge_case():
    """symbol_type_assist=False (the default) must never call the vision
    model, matching llm_ocr_assist's own opt-in-only contract."""
    shape = _shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm") as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8))

    mock_classify.assert_not_called()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "unknown"


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_unresolved_classification_leaves_shape_unknown_failure_path():
    """When the model can't confidently classify a shape either (both
    coarse/subtype come back None), the shape stays "unknown" -- same as if
    the flag were off, not a crash and not a fabricated guess."""
    shape = _shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm", return_value=(None, None, "unknown")),
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), symbol_type_assist=True)

    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "unknown"
