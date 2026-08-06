import numpy as np
import pytest

from src.pid_extraction import title_block
from src.pid_extraction.title_block import (
    PRESSURE_PATTERN,
    TEMPERATURE_PATTERN,
    _prefix_positions,
    extract_title_block_limits,
)

_DUMMY_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)


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


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_title_block_limits_missing_clause_leaves_unresolved_regression(monkeypatch):
    """Real bug: when OCR found fewer pressure/temperature clauses than tags,
    the positional-pairing fallback silently reused tag[0]'s clause for every
    tag past the matched count instead of leaving it unresolved -- misattributing
    one equipment's design limit to another. V-745 here has no stated clause at
    all and must resolve to None, not silently inherit F-715's 275 PSIG / 100F."""
    text = "F-715 A & B DESIGN: 275 PSIG @ 100F V-745 SEE NOTE"
    monkeypatch.setattr(title_block, "_ocr_header", lambda image: text)

    results = extract_title_block_limits(_DUMMY_IMAGE, known_tags={"F-715", "V-745"})

    assert results["F-715"]["design_pressure_psig"] == 275.0
    assert results["F-715"]["design_temperature_f"] == "100"
    assert results["V-745"]["design_pressure_psig"] is None
    assert results["V-745"]["design_temperature_f"] is None


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_title_block_limits_shared_prefix_tags_both_surface_regression(monkeypatch):
    """Real bug: known_prefixes used to be a plain {prefix: tag} dict, so two
    known tags sharing a prefix letter (e.g. two "MV-" tags) silently collapsed
    to whichever one the dict comprehension kept -- the other became permanently
    unmatchable via title-block OCR with no signal it happened. Both must now
    surface, sharing the one clause OCR actually found (a single header
    instance can't distinguish which physical tag it belongs to)."""
    text = "MV-701 DESIGN: 250 PSIG @ 90F"
    monkeypatch.setattr(title_block, "_ocr_header", lambda image: text)

    results = extract_title_block_limits(_DUMMY_IMAGE, known_tags={"MV-701", "MV-702"})

    assert set(results) == {"MV-701", "MV-702"}
    assert results["MV-701"]["design_pressure_psig"] == 250.0
    assert results["MV-702"]["design_pressure_psig"] == 250.0


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_extract_title_block_limits_two_tags_two_clauses_paired_by_position(monkeypatch):
    text = "AC-746 MWP 275 PSIG @ 100F E-742 MWP 300 PSIG @ 150F"
    monkeypatch.setattr(title_block, "_ocr_header", lambda image: text)

    results = extract_title_block_limits(_DUMMY_IMAGE, known_tags={"AC-746", "E-742"})

    assert results["AC-746"]["design_pressure_psig"] == 275.0
    assert results["E-742"]["design_pressure_psig"] == 300.0


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_extract_title_block_limits_no_known_tag_found_returns_empty(monkeypatch):
    monkeypatch.setattr(title_block, "_ocr_header", lambda image: "NOTHING RELEVANT HERE")
    assert extract_title_block_limits(_DUMMY_IMAGE, known_tags={"F-715"}) == {}
