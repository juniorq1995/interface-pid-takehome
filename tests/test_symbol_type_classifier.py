from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.shape_detection import DetectedShape
from src.pid_extraction.symbol_type_classifier import _parse_classification, classify_symbol_type_with_llm


def _shape():
    return DetectedShape(
        shape_id=0, kind="unknown", bbox=(10, 10, 40, 40), center=(30, 30), contour=np.empty((0, 1, 2), dtype=np.int32)
    )


def _image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def _mock_response(text: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": text}
    return resp


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_parse_classification_coarse_only_happy_path():
    assert _parse_classification("valve") == ("valve", None)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_parse_classification_coarse_and_known_subtype_happy_path():
    assert _parse_classification("valve, gate_valve") == ("valve", "gate_valve")


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_parse_classification_unknown_subtype_kept_coarse_only_edge_case():
    """A subtype the model invents that isn't in this project's vocabulary is
    dropped rather than trusted verbatim -- more likely hallucinated than a
    real gap in the list."""
    assert _parse_classification("instrument, some_made_up_gauge_type") == ("instrument", None)


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_parse_classification_off_vocabulary_coarse_returns_none_edge_case():
    assert _parse_classification("gauge") == (None, None)


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_parse_classification_empty_string_edge_case():
    assert _parse_classification("") == (None, None)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.symbol_type_classifier.requests.post")
def test_classify_symbol_type_with_llm_happy_path(mock_post):
    mock_post.return_value = _mock_response("valve, check_valve")
    coarse, subtype, raw = classify_symbol_type_with_llm(_image(), _shape())
    assert coarse == "valve"
    assert subtype == "check_valve"
    assert raw == "valve, check_valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.symbol_type_classifier.requests.post")
def test_classify_symbol_type_with_llm_explicit_unknown_edge_case(mock_post):
    mock_post.return_value = _mock_response("unknown")
    coarse, subtype, raw = classify_symbol_type_with_llm(_image(), _shape())
    assert coarse is None
    assert subtype is None
    assert raw == "unknown"


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.symbol_type_classifier.requests.post")
def test_classify_symbol_type_with_llm_network_failure_degrades_gracefully_failure_path(mock_post):
    import requests

    mock_post.side_effect = requests.exceptions.Timeout("model too slow")
    coarse, subtype, raw = classify_symbol_type_with_llm(_image(), _shape())
    assert (coarse, subtype, raw) == (None, None, "")


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_classify_symbol_type_with_llm_empty_crop_edge_case():
    """A shape whose bbox falls entirely outside the image (crop.size == 0)
    must not raise, and must not call the network."""
    shape = DetectedShape(
        shape_id=0, kind="unknown", bbox=(1000, 1000, 40, 40), center=(1020, 1020),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )
    with patch("src.pid_extraction.symbol_type_classifier.requests.post") as mock_post:
        coarse, subtype, raw = classify_symbol_type_with_llm(_image(), shape)
        mock_post.assert_not_called()
    assert (coarse, subtype, raw) == (None, None, "")
