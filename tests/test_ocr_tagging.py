"""Tests for src/pid_extraction/ocr_tagging.py.

No dedicated coverage existed for this module before this file -- extract_tag
was only ever exercised indirectly, patched out as a dependency in
test_pipeline_merge_shapes.py. Added while fixing a real gap found by direct
inspection of real failed valve-tag crops: most unread valve tags on this
document are rotated 90 degrees, and the original single-attempt --psm 7
call assumed horizontal text.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.ocr_tagging import extract_tag
from src.pid_extraction.shape_detection import DetectedShape


def _shape():
    return DetectedShape(
        shape_id=0, kind="bowtie", bbox=(10, 10, 40, 20), center=(30, 20), contour=np.empty((0, 1, 2), dtype=np.int32)
    )


def _image():
    return np.full((200, 200, 3), 255, dtype=np.uint8)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.ocr_tagging.pytesseract.image_to_string")
def test_extract_tag_reads_upright_text_on_first_attempt_happy_path(mock_ocr):
    mock_ocr.return_value = "MV-715-01"
    tag, raw = extract_tag(_image(), _shape())

    assert tag == "MV-715-01"
    assert raw == "MV-715-01"
    mock_ocr.assert_called_once()  # first rotation+psm combo already matched -- no wasted retries


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.ocr_tagging.pytesseract.image_to_string")
def test_extract_tag_recovers_tag_only_visible_after_rotation_regression(mock_ocr):
    """Regression for the real, diagnosed gap: a tag that only OCRs correctly
    once the crop is rotated (simulating rotated valve-tag text) must still
    be found, not just tried-and-discarded on the unrotated first attempt."""
    # 0deg psm7, 0deg psm11, 90CW psm7 all fail; 90CW psm11 finally succeeds.
    mock_ocr.side_effect = ["", "garbage", "", "MV-715-04B"]
    tag, raw = extract_tag(_image(), _shape())

    assert tag == "MV-715-04B"
    assert mock_ocr.call_count == 4


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.ocr_tagging.pytesseract.image_to_string")
def test_extract_tag_all_rotations_and_psms_fail_edge_case(mock_ocr):
    """No rotation/PSM combo produces a match -- must exhaust all 6 attempts
    (3 rotations x 2 PSM configs) and return None, not raise or give up early."""
    mock_ocr.return_value = "unreadable smudge"
    tag, raw = extract_tag(_image(), _shape())

    assert tag is None
    assert raw == "unreadable smudge"
    assert mock_ocr.call_count == 6


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_tag_empty_crop_returns_none_edge_case():
    """A shape whose bbox falls entirely outside the image must not raise,
    and must not call OCR at all."""
    shape = DetectedShape(
        shape_id=0, kind="bowtie", bbox=(1000, 1000, 40, 20), center=(1020, 1010),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )
    with patch("src.pid_extraction.ocr_tagging.pytesseract.image_to_string") as mock_ocr:
        tag, raw = extract_tag(_image(), shape)
        mock_ocr.assert_not_called()
    assert (tag, raw) == (None, "")


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.ocr_tagging.pytesseract.image_to_string")
def test_extract_tag_last_raw_text_reflects_most_recent_attempt_failure_path(mock_ocr):
    """On a full miss, raw_text should be whatever the last attempt actually
    read (useful for debugging), not the first attempt or a fixed sentinel."""
    mock_ocr.side_effect = ["first junk", "second junk", "third junk", "fourth junk", "fifth junk", "last junk"]
    tag, raw = extract_tag(_image(), _shape())

    assert tag is None
    assert raw == "last junk"
