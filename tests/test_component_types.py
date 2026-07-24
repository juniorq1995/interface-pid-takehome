import pytest

from src.component_types import coarse_category, type_from_tag


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_type_from_tag_happy_path():
    assert type_from_tag("V-101") == "valve"
    assert type_from_tag("FT-101") == "flow_transmitter"
    assert type_from_tag("PSV-501A") == "safety_valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@pytest.mark.multi_model  # same "unrecognized input -> None" invariant independently
# covered by the Opus review's test_type_from_tag_empty_and_separator_only_return_none
def test_type_from_tag_unknown_prefix_edge_case():
    assert type_from_tag("XYZ-999") is None


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_coarse_category_groups_instrument_variants():
    assert coarse_category("flow_transmitter") == "instrument"
    assert coarse_category("pressure_transmitter") == "instrument"
    assert coarse_category("valve") == "valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_coarse_category_none_input_edge_case():
    # No exception is raised for None — a graceful boundary case, not a
    # failure_path (that mark is reserved for tests asserting an error/raise).
    assert coarse_category(None) is None
