from src.pid_extraction.title_block import (
    PRESSURE_PATTERN,
    TEMPERATURE_PATTERN,
    _prefix_positions,
)


def test_prefix_positions_happy_path_clean_text():
    text = "F-715 A & B PARTICULATE FILTER"
    hits = _prefix_positions(text, {"F"})
    assert hits == [(0, "F")]


def test_prefix_positions_tolerates_ocr_corrupted_separator_edge_case():
    # Real OCR output seen on the assignment drawing: "-" misread as "=", "/", or an em-dash.
    for corrupted in ["F-/153 A & B", "F=715 A & B", "F—715 A & B"]:
        assert _prefix_positions(corrupted, {"F"}) == [(0, "F")], corrupted


def test_prefix_positions_does_not_collide_with_fahrenheit_unit_failure_case():
    # "F" as a bare temperature unit (no digit-bearing separator after it) must not match.
    text = "DESIGN: 275 PSIG @ 100 F"
    assert _prefix_positions(text, {"F"}) == []


def test_pressure_pattern_prefers_design_over_operating():
    text = "OPERATING: 230 PSIG DESIGN: 275 PSIG @ 100F"
    match = PRESSURE_PATTERN.search(text)
    assert match.group(1) == "275"


def test_temperature_pattern_matches_bare_number_before_f():
    text = "DESIGN: 275 PSIG @ 100F"
    match = TEMPERATURE_PATTERN.search(text)
    assert match.group(1) == "100"
