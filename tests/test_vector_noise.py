from pathlib import Path

import pytest

from scripts.generate_sample_data import PID_PATH as SYNTHETIC_PID_PATH
from src.pid_extraction.vector_noise import find_coil_regions_mediabox

REAL_PID_PATH = Path(__file__).resolve().parent.parent / "data" / "pid" / "diagram.pdf"
real_data_available = pytest.mark.skipif(not REAL_PID_PATH.exists(), reason="real assignment PDF not present")


@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_find_coil_regions_no_vector_data_edge_case():
    assert find_coil_regions_mediabox(SYNTHETIC_PID_PATH) == []


@pytest.mark.integration
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_find_coil_regions_missing_file_failure_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_coil_regions_mediabox(tmp_path / "nope.pdf")


@pytest.mark.regression
@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
@real_data_available
def test_find_coil_regions_finds_five_real_coils():
    """Pins the validated result: 5 coil/tubing symbols on page 0, each rendered
    and visually confirmed individually (see vector_noise.py's module docstring)
    before this count was trusted. Built to filter false candidates out of an
    exploratory valve-detection pass over pipe-segment endpoints — that
    downstream use didn't pan out reliably (see README "Closing the Valve Gap
    Further"), but this detector itself is real and independently useful."""
    regions = find_coil_regions_mediabox(REAL_PID_PATH, page_index=0)
    assert len(regions) == 5
    for r in regions:
        assert r.width > 0 and r.height > 0
