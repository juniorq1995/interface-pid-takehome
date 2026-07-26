from pathlib import Path

import pytest

from scripts.generate_sample_data import PID_PATH as SYNTHETIC_PID_PATH
from src.pid_extraction.vector_equipment import extract_vessel_symbols

REAL_PID_PATH = Path(__file__).resolve().parent.parent / "data" / "pid" / "diagram.pdf"
real_data_available = pytest.mark.skipif(not REAL_PID_PATH.exists(), reason="real assignment PDF not present")


@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_vessel_symbols_no_vector_data_edge_case():
    assert extract_vessel_symbols(SYNTHETIC_PID_PATH) == []


@pytest.mark.integration
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_extract_vessel_symbols_missing_file_failure_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_vessel_symbols(tmp_path / "nope.pdf")


@pytest.mark.regression
@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_extract_vessel_symbols_finds_both_known_vessels_real_pdf():
    """Pins the validated result: both F-715A and F-715B match the confirmed
    signature (60-90 line items, ~100x25pt) on page 0 — 100% equipment coverage
    for this page's 2 known vessels. See vector_equipment.py's module docstring
    for the two false leads (title-block logo, then a mistaken "confirmation"
    of a horizontal pipe stroke caused by over-generous crop padding) ruled out
    before this signature was trusted."""
    shapes = extract_vessel_symbols(REAL_PID_PATH, page_index=0, dpi=300)
    assert len(shapes) == 2
    for shape in shapes:
        assert shape.kind == "rectangle"
        _, _, w, h = shape.bbox
        assert h > w  # tall in rendered pixel space, matching a vertical vessel


@pytest.mark.regression
@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_extract_vessel_symbols_finds_v745_tower_top_on_page1():
    """V-745 (page 1) isn't one unified compound path like F-715 -- only its
    domed top + mesh-pad registers as a distinct compound path (48 items, tall/
    narrow, the opposite orientation of F-715's wide/short signature). Rendered
    tightly and visually confirmed before trusting; separately confirmed this
    signature produces zero matches on pages 0 and 2."""
    shapes = extract_vessel_symbols(REAL_PID_PATH, page_index=1, dpi=300)
    assert len(shapes) == 1
    _, _, w, h = shapes[0].bbox
    assert w > h  # wide in rendered pixel space -- the 270deg rotation swaps
    # the tall/narrow PDF-space fragment into a wide/short rendered fragment


@pytest.mark.regression
@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_extract_vessel_symbols_tower_signature_no_false_positives_page2():
    """The V-745 tower-top signature was calibrated against page 1 only --
    confirm it doesn't spuriously fire on unrelated page 2 content."""
    shapes = extract_vessel_symbols(REAL_PID_PATH, page_index=2, dpi=300)
    assert shapes == []
