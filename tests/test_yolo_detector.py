from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.yolo_detector import (
    class_name_to_component_type,
    detect_symbols,
    weights_available,
)


def _image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def _mock_box(x0, y0, x1, y1, class_id):
    box = MagicMock()
    box.xyxy = [MagicMock(tolist=lambda: [x0, y0, x1, y1])]
    box.cls = [MagicMock(item=lambda: class_id)]
    return box


def _mock_result(names, boxes):
    result = MagicMock()
    result.names = names
    result.boxes = boxes
    return result


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_class_name_to_component_type_maps_known_keyword():
    assert class_name_to_component_type("Gate Valve") == "valve"
    assert class_name_to_component_type("Pressure Transmitter") == "instrument"
    assert class_name_to_component_type("Horizontal Vessel") == "tank"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_class_name_to_component_type_unmapped_class_falls_back_to_unknown_edge_case():
    assert class_name_to_component_type("Not Gate") != "valve"
    assert class_name_to_component_type("Manhole") == "unknown"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_weights_available_true_when_file_exists(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    assert weights_available(weights) is True


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_weights_available_false_when_missing_edge_case(tmp_path):
    assert weights_available(tmp_path / "nope.pt") is False


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_detect_symbols_missing_weights_raises_failure_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_symbols(_image(), weights_path=tmp_path / "nope.pt")


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_detect_symbols_happy_path_returns_detected_shapes(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    names = {0: "Gate Valve", 1: "Pressure Transmitter"}
    boxes = [_mock_box(10, 20, 30, 40, 0), _mock_box(50, 60, 70, 90, 1)]
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_result(names, boxes)]

    with patch("ultralytics.YOLO", return_value=mock_model):
        shapes = detect_symbols(_image(), weights_path=weights)

    assert len(shapes) == 2
    assert shapes[0].kind == "gate_valve"
    assert shapes[0].bbox == (10, 20, 20, 20)
    assert shapes[0].center == (20, 30)
    assert shapes[1].kind == "pressure_transmitter"
    assert [s.shape_id for s in shapes] == [0, 1]


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_detect_symbols_no_detections_returns_empty_list_edge_case(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_result(names={}, boxes=[])]

    with patch("ultralytics.YOLO", return_value=mock_model):
        shapes = detect_symbols(_image(), weights_path=weights)

    assert shapes == []
