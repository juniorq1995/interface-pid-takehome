from src.component_types import coarse_category, type_from_tag


def test_type_from_tag_happy_path():
    assert type_from_tag("V-101") == "valve"
    assert type_from_tag("FT-101") == "flow_transmitter"
    assert type_from_tag("PSV-501A") == "safety_valve"


def test_type_from_tag_unknown_prefix_edge_case():
    assert type_from_tag("XYZ-999") is None


def test_coarse_category_groups_instrument_variants():
    assert coarse_category("flow_transmitter") == "instrument"
    assert coarse_category("pressure_transmitter") == "instrument"
    assert coarse_category("valve") == "valve"


def test_coarse_category_none_input_failure_case():
    assert coarse_category(None) is None
