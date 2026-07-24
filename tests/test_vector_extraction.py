from pathlib import Path

import pytest

from scripts.generate_sample_data import PID_PATH as SYNTHETIC_PID_PATH
from src.pid_extraction.vector_lines import extract_line_segments
from src.pid_extraction.vector_symbols import extract_circle_symbols

REAL_PID_PATH = Path(__file__).resolve().parent.parent / "data" / "pid" / "diagram.pdf"
real_data_available = pytest.mark.skipif(not REAL_PID_PATH.exists(), reason="real assignment PDF not present")


def test_extract_line_segments_no_vector_data_edge_case():
    # The synthetic fixture is a rasterized image embedded in a PDF — no vector
    # path data at all. Callers are expected to fall back to raster detection.
    assert extract_line_segments(SYNTHETIC_PID_PATH) == []


def test_extract_circle_symbols_no_vector_data_edge_case():
    assert extract_circle_symbols(SYNTHETIC_PID_PATH) == []


def test_extract_line_segments_missing_file_failure_case(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_line_segments(tmp_path / "nope.pdf")


def test_extract_circle_symbols_missing_file_failure_case(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_circle_symbols(tmp_path / "nope.pdf")


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


@real_data_available
def test_extract_circle_symbols_happy_path_real_pdf():
    shapes = extract_circle_symbols(REAL_PID_PATH, page_index=0, dpi=300)
    assert len(shapes) > 0
    for shape in shapes:
        assert shape.kind == "circle"
        _, _, w, h = shape.bbox
        assert 0.7 <= w / h <= 1.4  # near-square, matches the calibrated symbol band
