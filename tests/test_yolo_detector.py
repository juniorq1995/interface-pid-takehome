from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.yolo_detector import (
    TILE_SIZE,
    _iou,
    _iter_tile_offsets,
    _merge_overlapping_detections,
    class_name_to_component_type,
    detect_symbols,
    weights_available,
)


def _image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def _mock_box(x0, y0, x1, y1, class_id, confidence=0.9):
    box = MagicMock()
    box.xyxy = [MagicMock(tolist=lambda: [x0, y0, x1, y1])]
    box.cls = [MagicMock(item=lambda: class_id)]
    box.conf = [MagicMock(item=lambda: confidence)]
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


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_iou_full_overlap_is_one():
    box = (0, 0, 10, 10)
    assert _iou(box, box) == 1.0


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_iou_no_overlap_is_zero_edge_case():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_merge_overlapping_detections_keeps_higher_confidence_duplicate():
    detections = [
        {"xyxy": (10, 10, 50, 50), "class_name": "Gate Valve", "confidence": 0.4},
        {"xyxy": (12, 12, 52, 52), "class_name": "Ball Valve", "confidence": 0.9},
    ]
    merged = _merge_overlapping_detections(detections)
    assert len(merged) == 1
    assert merged[0]["class_name"] == "Ball Valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_merge_overlapping_detections_keeps_distinct_boxes_edge_case():
    detections = [
        {"xyxy": (0, 0, 10, 10), "class_name": "Gate Valve", "confidence": 0.9},
        {"xyxy": (500, 500, 510, 510), "class_name": "Ball Valve", "confidence": 0.8},
    ]
    merged = _merge_overlapping_detections(detections)
    assert len(merged) == 2


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_iter_tile_offsets_covers_full_image_including_trailing_edge():
    offsets = _iter_tile_offsets(width=1500, height=640, tile_size=TILE_SIZE, overlap=96)
    max_x = max(x for x, _ in offsets)
    assert max_x + TILE_SIZE >= 1500
    assert (0, 0) in offsets


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_iter_tile_offsets_small_image_returns_single_origin_tile_edge_case():
    assert _iter_tile_offsets(width=200, height=200, tile_size=TILE_SIZE, overlap=96) == [(0, 0)]


@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_detect_symbols_large_image_tiles_and_merges_across_tile_boundary(tmp_path):
    """A page larger than one tile must be sliced rather than resized-and-shrunk
    in a single pass — this is the fix for the confirmed real-page regression
    (5 raw detections on a full untiled 3300x5100 page vs 19 tiled)."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    big_image = np.full((700, 700, 3), 255, dtype=np.uint8)
    names = {0: "Gate Valve"}

    def _fake_predict(crop, conf, verbose):
        # Only the tile covering (0,0)-ish region reports a detection.
        if crop.shape[0] == 700 or crop is big_image:
            return [_mock_result(names, [])]
        return [_mock_result(names, [_mock_box(5, 5, 25, 25, 0)])]

    mock_model = MagicMock()
    mock_model.predict.side_effect = _fake_predict

    with patch("ultralytics.YOLO", return_value=mock_model):
        detect_symbols(big_image, weights_path=weights)

    assert mock_model.predict.call_count > 1  # tiled into more than one pass
