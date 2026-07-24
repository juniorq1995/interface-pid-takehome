import pytest

from src.pid_extraction.title_block import (
    PRESSURE_PATTERN,
    TEMPERATURE_PATTERN,
    _prefix_positions,
)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_prefix_positions_happy_path_clean_text():
    text = "F-715 A & B PARTICULATE FILTER"
    hits = _prefix_positions(text, {"F"})
    assert hits == [(0, "F")]


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_prefix_positions_tolerates_ocr_corrupted_separator_edge_case():
    # Real OCR output seen on the assignment drawing: "-" misread as "=", "/", or an em-dash.
    for corrupted in ["F-/153 A & B", "F=715 A & B", "F—715 A & B"]:
        assert _prefix_positions(corrupted, {"F"}) == [(0, "F")], corrupted


@pytest.mark.adversarial
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_prefix_positions_does_not_collide_with_fahrenheit_unit_adversarial():
    # Deliberately confusable input: "F" as a bare temperature unit (no
    # digit-bearing separator after it) must not false-positive match.
    text = "DESIGN: 275 PSIG @ 100 F"
    assert _prefix_positions(text, {"F"}) == []


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_pressure_pattern_prefers_design_over_operating():
    text = "OPERATING: 230 PSIG DESIGN: 275 PSIG @ 100F"
    match = PRESSURE_PATTERN.search(text)
    assert match.group(1) == "275"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_temperature_pattern_matches_bare_number_before_f():
    text = "DESIGN: 275 PSIG @ 100F"
    match = TEMPERATURE_PATTERN.search(text)
    assert match.group(1) == "100"
