import pytest

from src.crossref.limit_check import cross_reference_limits
from src.sop_extraction.limits_table import EquipmentLimit


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_cross_reference_limits_happy_path_no_mismatch():
    pid_limits = {"F-715": {"design_pressure_psig": 275.0, "design_temperature_f": "100"}}
    sop_limits = {"F-715": [EquipmentLimit("F-715", "F-715 Filters", 275.0, "100")]}

    assert cross_reference_limits(pid_limits, sop_limits) == []


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_cross_reference_limits_tolerates_ocr_noise_in_temperature_edge_case():
    # OCR read "3/75" for "375" — digit sequence intact, punctuation corrupted.
    pid_limits = {"V-745": {"design_pressure_psig": 300.0, "design_temperature_f": "3/75"}}
    sop_limits = {"V-745": [EquipmentLimit("V-745", "V-745 Stabilizer", 300.0, "375")]}

    assert cross_reference_limits(pid_limits, sop_limits) == []


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_cross_reference_limits_detects_real_pressure_mismatch():
    pid_limits = {"F-715": {"design_pressure_psig": 200.0, "design_temperature_f": "100"}}
    sop_limits = {"F-715": [EquipmentLimit("F-715", "F-715 Filters", 275.0, "100")]}

    findings = cross_reference_limits(pid_limits, sop_limits)
    assert len(findings) == 1
    assert findings[0].category == "DESIGN_LIMIT_MISMATCH"
    assert findings[0].severity == "ERROR"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_cross_reference_limits_matches_against_any_row_for_multi_row_tag():
    # E-742 has separate shell/tube rows — P&ID's single reading should match whichever applies.
    pid_limits = {"E-742": {"design_pressure_psig": 300.0, "design_temperature_f": "250"}}
    sop_limits = {
        "E-742": [
            EquipmentLimit("E-742", "E-742 Shell", 300.0, "375"),
            EquipmentLimit("E-742", "E-742 Tube", 300.0, "250"),
        ]
    }

    assert cross_reference_limits(pid_limits, sop_limits) == []


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_cross_reference_limits_no_sop_row_for_tag_edge_case():
    pid_limits = {"P-745": {"design_pressure_psig": 300.0, "design_temperature_f": "100"}}
    sop_limits = {}  # P-745 never appeared in the SOP table

    assert cross_reference_limits(pid_limits, sop_limits) == []
