"""Tests for src/pid_extraction/equipment_label_discovery.py.

Real gap this closes: pump/exchanger/cooler symbols have no clean geometric
signature (confirmed live -- zero shapes from any existing detector fall
within 95px of P-745's real location on page 1). This module finds them by
their text label instead of their shape.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.equipment_label_discovery import find_equipment_label_candidates
from src.pid_extraction.shape_detection import DetectedShape


def _image(w=3500, h=3500):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _ocr_data(words: list[tuple[str, int, int, int, int]]) -> dict:
    """words: list of (text, left, top, width, height)."""
    return {
        "text": [w[0] for w in words],
        "left": [w[1] for w in words],
        "top": [w[2] for w in words],
        "width": [w[3] for w in words],
        "height": [w[4] for w in words],
    }


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_finds_pump_label_happy_path(mock_ocr):
    mock_ocr.return_value = _ocr_data([("P-745", 1000, 2900, 60, 20)])
    candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[])

    assert len(candidates) == 1
    assert candidates[0].kind == "equipment_label_region"
    assert candidates[0].center == (1030, 2910)


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_finds_pump_label_with_hyphen_read_as_equals_sign_regression(mock_ocr):
    """Regression for the real, confirmed-live OCR quirk: Tesseract reads
    this document's tag hyphens as "=" often enough that a naive hyphen-only
    tag pattern missed P-745's real label entirely on first attempt."""
    mock_ocr.return_value = _ocr_data([("P=745", 1000, 2900, 60, 20)])
    candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[])

    assert len(candidates) == 1


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_ignores_non_equipment_prefixes_edge_case():
    """MV (valve) and PI (instrument) prefixes must not be treated as
    equipment discoveries -- this module is equipment-only by design."""
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        mock_ocr.return_value = _ocr_data([("MV-715-01", 500, 500, 60, 20), ("PI-101", 700, 700, 50, 20)])
        candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[])

    assert candidates == []


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_suppresses_label_near_tall_existing_vessel_regression():
    """Regression: a naive center-distance exclusion check missed F-715A's
    own label on real data, because a tall vessel's (101x437px) geometric
    center is far from where its label actually sits near the top -- wrongly
    re-discovering an already-detected vessel as a "new" candidate. The
    bbox-containment-with-margin check must suppress this."""
    # Real F-715A vessel bbox and real label position relative to it.
    vessel = DetectedShape(
        shape_id=0, kind="rectangle", bbox=(2010, 1600, 101, 437), center=(2061, 1818),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        mock_ocr.return_value = _ocr_data([("F-715A", 1880, 1610, 60, 20)])  # near the top, left of bbox
        candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[vessel])

    assert candidates == []


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_does_not_suppress_unrelated_equipment_far_from_existing_vessel_happy_path():
    vessel = DetectedShape(
        shape_id=0, kind="rectangle", bbox=(2010, 1600, 101, 437), center=(2061, 1818),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        mock_ocr.return_value = _ocr_data([("P-745", 1000, 2900, 60, 20)])  # far from the vessel
        candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[vessel])

    assert len(candidates) == 1


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_dedups_repeated_label_detection_at_same_position_edge_case():
    """Tesseract can emit near-duplicate word fragments at nearly the same
    position -- must not produce two candidates for one real label."""
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        mock_ocr.return_value = _ocr_data([
            ("P-745", 1000, 2900, 60, 20),
            ("P-745", 1005, 2903, 60, 20),  # same label, tiny jitter
        ])
        candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[])

    assert len(candidates) == 1


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_start_shape_id_offsets_candidate_ids_edge_case():
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        mock_ocr.return_value = _ocr_data([("P-745", 1000, 2900, 60, 20)])
        candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[], start_shape_id=50)

    assert candidates[0].shape_id == 50


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_no_ocr_hits_returns_empty_list_failure_path():
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        mock_ocr.return_value = _ocr_data([("", 0, 0, 0, 0), ("random text", 10, 10, 40, 10)])
        candidates = find_equipment_label_candidates(_image(), existing_equipment_shapes=[])

    assert candidates == []
