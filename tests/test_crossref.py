import networkx as nx
import pytest

from src.crossref.compare import cross_reference
from src.sop_extraction.tag_extractor import SopFacts


def _node(graph, tag, component_type):
    graph.add_node(tag, tag=tag, component_type=component_type, symbol_kind="rectangle", ocr_raw_text=tag, bbox=[0, 0, 1, 1])


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_cross_reference_happy_path_no_discrepancies():
    graph = nx.Graph()
    _node(graph, "T-101", "tank")
    _node(graph, "P-101", "pump")
    graph.add_edge("T-101", "P-101")

    facts = SopFacts(
        referenced_tags={"T-101", "P-101"},
        declared_types={"T-101": "tank", "P-101": "pump"},
        stated_connections={("P-101", "T-101")},
    )

    findings = cross_reference(graph, facts)
    assert findings == []


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_cross_reference_missing_in_pid_edge_case():
    graph = nx.Graph()
    facts = SopFacts(referenced_tags={"PSV-201"})

    findings = cross_reference(graph, facts)
    assert len(findings) == 1
    assert findings[0].category == "MISSING_IN_PID"
    assert findings[0].severity == "ERROR"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_cross_reference_type_and_connection_mismatch():
    graph = nx.Graph()
    _node(graph, "T-101", "tank")
    _node(graph, "FT-101", "flow_transmitter")
    # no edge between them

    facts = SopFacts(
        referenced_tags={"T-101", "FT-101"},
        declared_types={"T-101": "pump"},
        stated_connections={("FT-101", "T-101")},
    )

    findings = cross_reference(graph, facts)
    categories = {f.category for f in findings}
    assert "TYPE_MISMATCH" in categories
    assert "CONNECTION_MISMATCH" in categories


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_cross_reference_unresolved_tag_edge_case():
    graph = nx.Graph()
    graph.add_node("UNLABELED-0", tag=None, component_type="unknown", symbol_kind="rectangle", ocr_raw_text="", bbox=[0, 0, 1, 1])

    findings = cross_reference(graph, SopFacts())
    assert len(findings) == 1
    assert findings[0].category == "UNRESOLVED_TAG"
    assert findings[0].severity == "WARNING"
