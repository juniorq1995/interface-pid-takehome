"""Tests for src/pid_extraction/equipment_label_discovery.py.

Real gap this closes: pump/exchanger/cooler symbols have no clean geometric
signature (confirmed live -- zero shapes from any existing detector fall
within 95px of P-745's real location on page 1). This module finds them by
their text label instead of their shape.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.equipment_label_discovery import find_equipment_label_candidates, find_tag_for_known_equipment_shape
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


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_finds_air_cooler_label_with_hyphen_read_as_em_dash_regression(mock_ocr):
    """Regression for a second, distinct real corrupted-separator case found
    diagnosing AC-746 specifically: Tesseract read this label's hyphen as an
    em-dash ("AC—746"), not the "=" the P-745 regression above covers --
    confirmed live via direct OCR on the real page-2 crop. AC-746 was a
    silent, un-diagnosed false negative until this was found: it never even
    produced a wrong tag, because no candidate region was ever created for
    it at all."""
    mock_ocr.return_value = _ocr_data([("AC—746", 1000, 2900, 60, 20)])
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


def _vessel_shape():
    """Mirrors F-715A's real bbox: 101x437px, well beyond the label-crop
    size guard."""
    return DetectedShape(
        shape_id=0, kind="rectangle", bbox=(2010, 1600, 101, 437), center=(2061, 1818),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_find_tag_for_known_equipment_shape_happy_path(mock_ocr):
    mock_ocr.return_value = _ocr_data([("F-715A", 200, 200, 60, 20)])
    tag = find_tag_for_known_equipment_shape(_image(), _vessel_shape())

    assert tag == "F-715A"


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_find_tag_for_known_equipment_shape_normalizes_corrupted_separator_regression(mock_ocr):
    """Regression for the same real, confirmed-live OCR quirk as the
    discovery-candidate path: Tesseract reads this document's tag hyphens as
    "=" often enough to matter. Unlike discovery, this path must produce the
    real tag *text* (for tag-accuracy scoring), so the corrupted separator is
    normalized back to "-" rather than just used to find a location."""
    mock_ocr.return_value = _ocr_data([("F=715A", 200, 200, 60, 20)])
    tag = find_tag_for_known_equipment_shape(_image(), _vessel_shape())

    assert tag == "F-715A"


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_find_tag_for_known_equipment_shape_normalizes_em_dash_regression(mock_ocr):
    """Regression for AC-746's real corrupted-separator case: em-dash, not
    "=". Same normalize-then-strict-match path as the "=" case above, a
    distinct corruption confirmed live on the real document."""
    mock_ocr.return_value = _ocr_data([("AC—746", 200, 200, 60, 20)])
    tag = find_tag_for_known_equipment_shape(_image(), _vessel_shape())

    assert tag == "AC-746"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_find_tag_for_known_equipment_shape_ignores_non_equipment_prefix_edge_case(mock_ocr):
    """A nearby valve tag must not be picked up as the equipment's own tag --
    this is exactly the class of bug the oversized-crop guard exists to
    prevent; this targeted search must not reintroduce it."""
    mock_ocr.return_value = _ocr_data([("MV-715-01", 200, 200, 60, 20)])
    tag = find_tag_for_known_equipment_shape(_image(), _vessel_shape())

    assert tag is None


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data")
def test_find_tag_for_known_equipment_shape_picks_nearest_of_multiple_happy_path(mock_ocr):
    # Search crop is 600x600 (300px radius), so the shape's own center sits
    # at crop-local (300, 300) when nothing clamps against the image edge.
    mock_ocr.return_value = _ocr_data([
        ("F-999Z", 10, 10, 60, 20),  # far corner of the search crop
        ("F-715A", 290, 290, 60, 20),  # right next to the shape's actual center
    ])
    tag = find_tag_for_known_equipment_shape(_image(3500, 3500), _vessel_shape())

    assert tag == "F-715A"


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_find_tag_for_known_equipment_shape_empty_crop_returns_none_failure_path():
    """A shape whose search region falls entirely outside the image must not
    raise, and must not call OCR at all."""
    shape = DetectedShape(
        shape_id=0, kind="rectangle", bbox=(10000, 10000, 101, 437), center=(10050, 10200),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )
    with patch("src.pid_extraction.equipment_label_discovery.pytesseract.image_to_data") as mock_ocr:
        tag = find_tag_for_known_equipment_shape(_image(), shape)
        mock_ocr.assert_not_called()
    assert tag is None
