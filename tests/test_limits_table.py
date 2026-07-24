import pytest

from src.sop_extraction.limits_table import extract_equipment_limits


def _table():
    return [
        ["", "Pressure (psig)", "Temperature (°F)"],
        ["F-715 A and B Particulate Filters", "275", "100"],
        ["E-742 Exchanger (Shell)", "300", "375"],
        ["E-742 Exchanger (Tube)", "300", "250"],
    ]


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_extract_equipment_limits_happy_path():
    limits = extract_equipment_limits([_table()])

    assert limits["F-715"][0].pressure_psig == 275.0
    assert limits["F-715"][0].temperature_f == "100"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_equipment_limits_keeps_multiple_rows_per_tag_edge_case():
    limits = extract_equipment_limits([_table()])

    assert len(limits["E-742"]) == 2
    temps = {row.temperature_f for row in limits["E-742"]}
    assert temps == {"375", "250"}


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_equipment_limits_skips_table_without_pressure_or_temp_columns_edge_case():
    other_table = [["Name", "Notes"], ["F-715", "some note"]]
    limits = extract_equipment_limits([other_table])
    assert limits == {}


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_equipment_limits_empty_input_edge_case():
    assert extract_equipment_limits([]) == {}
    assert extract_equipment_limits([[]]) == {}
