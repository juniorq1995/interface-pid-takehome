from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pid_extraction.llm_ocr_assist import read_tag_with_llm
from src.pid_extraction.shape_detection import DetectedShape


def _shape():
    return DetectedShape(
        shape_id=0, kind="circle", bbox=(10, 10, 40, 40), center=(30, 30), contour=np.empty((0, 1, 2), dtype=np.int32)
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
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_happy_path(mock_post):
    mock_post.return_value = _mock_response("PSV 715B")
    tag, raw = read_tag_with_llm(_image(), _shape())
    assert tag == "PSV 715B"
    assert raw == "PSV 715B"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_explicit_unknown_edge_case(mock_post):
    mock_post.return_value = _mock_response("UNKNOWN")
    tag, raw = read_tag_with_llm(_image(), _shape())
    assert tag is None
    assert raw == "UNKNOWN"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_unparseable_response_edge_case(mock_post):
    mock_post.return_value = _mock_response("I see a small circle symbol but no legible text.")
    tag, raw = read_tag_with_llm(_image(), _shape())
    assert tag is None
    assert raw  # raw model text is still preserved for debugging


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_network_failure_degrades_gracefully_failure_path(mock_post):
    import requests

    mock_post.side_effect = requests.exceptions.Timeout("model too slow")
    tag, raw = read_tag_with_llm(_image(), _shape())
    assert (tag, raw) == (None, "")
