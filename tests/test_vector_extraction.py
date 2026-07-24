from pathlib import Path

import pytest

from scripts.generate_sample_data import PID_PATH as SYNTHETIC_PID_PATH
from src.pid_extraction.vector_lines import extract_line_segments
from src.pid_extraction.vector_symbols import extract_circle_symbols
from src.pid_extraction.vector_valves import extract_valve_symbols

REAL_PID_PATH = Path(__file__).resolve().parent.parent / "data" / "pid" / "diagram.pdf"
real_data_available = pytest.mark.skipif(not REAL_PID_PATH.exists(), reason="real assignment PDF not present")


@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_line_segments_no_vector_data_edge_case():
    # The synthetic fixture is a rasterized image embedded in a PDF — no vector
    # path data at all. Callers are expected to fall back to raster detection.
    assert extract_line_segments(SYNTHETIC_PID_PATH) == []


@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_circle_symbols_no_vector_data_edge_case():
    assert extract_circle_symbols(SYNTHETIC_PID_PATH) == []


@pytest.mark.integration
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_extract_line_segments_missing_file_failure_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_line_segments(tmp_path / "nope.pdf")


@pytest.mark.integration
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_extract_circle_symbols_missing_file_failure_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_circle_symbols(tmp_path / "nope.pdf")


@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_valve_symbols_no_vector_data_edge_case():
    assert extract_valve_symbols(SYNTHETIC_PID_PATH) == []


@pytest.mark.integration
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_extract_valve_symbols_missing_file_failure_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_valve_symbols(tmp_path / "nope.pdf")


@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_extract_line_segments_happy_path_real_pdf():
    segments = extract_line_segments(REAL_PID_PATH, page_index=0, dpi=300)
    assert len(segments) > 100  # real sheet has hundreds of pipe/border strokes

    # None of the kept segments should be the page border itself (filtered by
    # MAX_LENGTH_FRACTION_OF_PAGE) — border segments run corner-to-corner at
    # roughly the full page width/height in pixel space at this DPI.
    page_width_px = 1224 * 300 / 72
    for (x1, y1), (x2, y2) in segments:
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        assert length < 0.6 * page_width_px


@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_extract_circle_symbols_happy_path_real_pdf():
    shapes = extract_circle_symbols(REAL_PID_PATH, page_index=0, dpi=300)
    assert len(shapes) > 0
    for shape in shapes:
        assert shape.kind == "circle"
        _, _, w, h = shape.bbox
        assert 0.7 <= w / h <= 1.4  # near-square, matches the calibrated symbol band


@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_extract_valve_symbols_partial_recall_real_pdf():
    # ~10/29 known valves on this page match the two confirmed vector signatures
    # (isolated bowtie + fused pipe-valve — see README "Vector-Based Valve
    # Detection"). This test pins the honest partial-recall count, not a claim
    # of full coverage. Fused-signature matches are the full pipe run's bbox,
    # not a tight square around the bowtie, so aspect ratio isn't constrained
    # the way the isolated-bowtie signature's near-square matches are.
    shapes = extract_valve_symbols(REAL_PID_PATH, page_index=0, dpi=300)
    assert 1 <= len(shapes) <= 20
    for shape in shapes:
        assert shape.kind == "bowtie"
