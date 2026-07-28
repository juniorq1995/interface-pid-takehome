import pytest

from src.component_types import coarse_category, project_coarse_category, type_from_tag


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


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_project_coarse_category_collapses_fine_instrument_subtypes_to_instrument():
    """Regression: score_cv_accuracy.py used to carry its own 4-entry local
    coarse map that didn't cover most of TAG_PREFIX_TYPES' ~44 fine-grained
    values -- a correctly tag-resolved "pressure_indicator" or
    "pressure_differential_indicator" (etc.) fell through it and vanished
    from the "instrument" category-coverage row entirely once
    llm_ocr_assist started resolving real tags. This is the shared,
    complete replacement."""
    assert project_coarse_category("pressure_indicator") == "instrument"
    assert project_coarse_category("pressure_differential_indicator") == "instrument"
    assert project_coarse_category("hand_off_auto_switch") == "instrument"
    assert project_coarse_category("thermowell") == "instrument"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_project_coarse_category_collapses_valve_and_equipment_variants():
    assert project_coarse_category("safety_valve") == "valve"
    assert project_coarse_category("pressure_control_valve") == "valve"
    assert project_coarse_category("emergency_shutdown_valve") == "valve"
    assert project_coarse_category("tank") == "equipment"
    assert project_coarse_category("pump") == "equipment"
    assert project_coarse_category("compressor") == "equipment"
    assert project_coarse_category("heat_exchanger") == "equipment"
    assert project_coarse_category("filter") == "equipment"


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_project_coarse_category_accepts_literal_equipment_identity_regression():
    """Regression: equipment_label_discovery.py's candidates are typed
    directly as the literal coarse name "equipment" (no fine-grained subtype
    to assign). Without this, project_coarse_category("equipment") fell
    through to the default "instrument" branch -- measured live: page 1
    instrument count inflated 29 -> 32 by the very candidates meant to fix
    the equipment gap, while equipment coverage never moved."""
    assert project_coarse_category("equipment") == "equipment"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_project_coarse_category_none_and_unknown_edge_case():
    assert project_coarse_category(None) is None
    assert project_coarse_category("unknown") is None


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_project_coarse_category_differs_from_coarse_category_for_equipment_subtypes_edge_case():
    """The two functions are deliberately different: coarse_category() keeps
    "tank"/"pump" un-collapsed for crossref/compare.py's SOP declared_types
    comparison; project_coarse_category() collapses them to "equipment" for
    this project's own 3-category ground truth. Pinning the divergence so a
    future "simplification" doesn't accidentally merge them back together."""
    assert coarse_category("tank") == "tank"
    assert project_coarse_category("tank") == "equipment"
