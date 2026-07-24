from pathlib import Path

import pytest

from scripts.generate_sample_data import PID_PATH, SOP_PATH
from src.crossref.compare import cross_reference
from src.pid_extraction.pipeline import extract_pid_graph
from src.sop_extraction.docx_parser import parse_sop
from src.sop_extraction.tag_extractor import extract_sop_facts


@pytest.mark.pipeline
@pytest.mark.integration
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_full_pipeline_happy_path_finds_designed_discrepancies():
    graph = extract_pid_graph(PID_PATH)
    sop = parse_sop(SOP_PATH)
    facts = extract_sop_facts(sop.full_text)
    findings = cross_reference(graph, facts)

    categories = {f.category for f in findings}
    # The synthetic fixture (scripts/generate_sample_data.py) is deliberately
    # built to exercise every discrepancy type — see its module docstring.
    assert categories == {
        "MISSING_IN_PID",
        "MISSING_IN_SOP",
        "TYPE_MISMATCH",
        "CONNECTION_MISMATCH",
        "UNRESOLVED_TAG",
    }

    # The one SOP-stated connection that matches a real P&ID edge (P-101 <-> V-101)
    # must NOT be flagged — proves true positives aren't drowned out by false ones.
    messages = " ".join(f.message for f in findings)
    assert "P-101 connects to V-101" not in messages


@pytest.mark.regression
@pytest.mark.integration
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_pipeline_real_edges_match_designed_chain():
    """Regression: an earlier connector_detection.py implementation sampled straight-line
    "ink coverage" between every pair of shape centers. On this fixture's inline chain
    (T-101 - P-101 - V-101 - FT-101), that produced a spurious direct T-101<->FT-101 edge,
    because the line between their centers passes straight through the two shapes between
    them and reads as continuous ink. Fixed by masking symbol interiors and using Hough
    line segments attached to their nearest symbol instead of raw center-to-center ink
    sampling (see connector_detection.py docstring). This test pins that fix."""
    graph = extract_pid_graph(PID_PATH)
    assert graph.has_edge("T-101", "P-101")
    assert graph.has_edge("P-101", "V-101")
    assert graph.has_edge("V-101", "FT-101")
    assert not graph.has_edge("T-101", "FT-101")  # would be the false "skip-ahead" edge


@pytest.mark.integration
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_extract_pid_graph_missing_file_failure_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_pid_graph(tmp_path / "nope.pdf")
