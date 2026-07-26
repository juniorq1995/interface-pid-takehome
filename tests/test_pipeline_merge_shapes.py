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
