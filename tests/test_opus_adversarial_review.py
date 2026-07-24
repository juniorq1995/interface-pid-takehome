"""Independent second-reviewer adversarial/edge pass (Claude Opus).

Targets high-stakes discrepancy-detection logic: compare.py, limit_check.py,
component_types.py, llm_ocr_assist.py, tag_extractor.py. Focus is on inputs a
first author plausibly wouldn't try: duplicate/whitespace/case-variant tags,
floating-point tolerance boundaries, digit-concatenated temperature ranges,
malformed HTTP responses, and chained SOP connection phrases.
"""
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from src.component_types import type_from_tag
from src.crossref.compare import cross_reference
from src.crossref.limit_check import cross_reference_limits
from src.pid_extraction.llm_ocr_assist import read_tag_with_llm
from src.pid_extraction.shape_detection import DetectedShape
from src.sop_extraction.limits_table import EquipmentLimit
from src.sop_extraction.tag_extractor import SopFacts, extract_sop_facts


def _node(graph, label, tag, component_type="tank"):
    graph.add_node(label, tag=tag, component_type=component_type, symbol_kind="rectangle", ocr_raw_text=tag or "", bbox=[0, 0, 1, 1])


def _shape():
    return DetectedShape(shape_id=0, kind="circle", bbox=(10, 10, 40, 40), center=(30, 30), contour=np.empty((0, 1, 2), dtype=np.int32))


def _image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def _mock_response(text):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": text}
    return resp


# ---------------- compare.cross_reference ----------------

@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_cross_reference_duplicate_tag_nodes_collapse_silently():
    # Two distinct symbols both OCR'd to the same tag: _graph_tags is keyed by
    # tag, so the second silently overwrites the first — only one finding emitted.
    graph = nx.Graph()
    _node(graph, "sym1", "P-101", "pump")
    _node(graph, "sym2", "P-101", "pump")

    findings = cross_reference(graph, SopFacts())
    missing_in_sop = [f for f in findings if f.category == "MISSING_IN_SOP"]
    assert len(missing_in_sop) == 1  # the duplicate is lost, not flagged as a collision


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model
def test_cross_reference_is_case_sensitive_on_tags():
    graph = nx.Graph()
    _node(graph, "t-101", "t-101", "tank")  # lowercase in graph
    facts = SopFacts(referenced_tags={"T-101"})  # uppercase in SOP

    cats = {f.category for f in cross_reference(graph, facts)}
    assert "MISSING_IN_PID" in cats   # SOP's T-101 not found
    assert "MISSING_IN_SOP" in cats   # graph's t-101 not found — same physical tag, double-counted


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_cross_reference_whitespace_padded_tag_treated_as_distinct():
    graph = nx.Graph()
    _node(graph, "T-101 ", "T-101 ", "tank")  # trailing space from OCR
    facts = SopFacts(referenced_tags={"T-101"})

    cats = {f.category for f in cross_reference(graph, facts)}
    assert "MISSING_IN_PID" in cats and "MISSING_IN_SOP" in cats


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_cross_reference_node_without_component_type_skips_type_check():
    graph = nx.Graph()
    graph.add_node("T-101", tag="T-101", symbol_kind="rectangle", ocr_raw_text="T-101", bbox=[0, 0, 1, 1])  # no component_type
    facts = SopFacts(referenced_tags={"T-101"}, declared_types={"T-101": "pump"})

    cats = {f.category for f in cross_reference(graph, facts)}
    assert "TYPE_MISMATCH" not in cats  # coarse_category(None) -> None -> no false mismatch


# ---------------- limit_check.cross_reference_limits ----------------

@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_limit_check_floating_point_boundary_fixed_at_exact_half_psig():
    """Regression: 127.8 and 128.3 are exactly 0.5 psig apart (== tolerance, should
    match), but raw float subtraction yields 0.5000000000000142 > 0.5, spuriously
    flagging it. Found by independent review; fixed in limit_check.py by rounding
    the diff to FLOAT_COMPARISON_DECIMALS before comparing against the tolerance."""
    pid_limits = {"F-715": {"design_pressure_psig": 127.8, "design_temperature_f": "100"}}
    sop_limits = {"F-715": [EquipmentLimit("F-715", "F-715 Filters", 128.3, "100")]}

    assert cross_reference_limits(pid_limits, sop_limits) == []


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model
def test_limit_check_zero_pressure_is_checked_not_treated_as_missing():
    # 0.0 is falsy but not None — a naive `if not pid_value` would skip it.
    pid_limits = {"P-1": {"design_pressure_psig": 0.0, "design_temperature_f": "100"}}
    sop_limits = {"P-1": [EquipmentLimit("P-1", "P-1", 300.0, "100")]}

    findings = cross_reference_limits(pid_limits, sop_limits)
    assert [f.category for f in findings] == ["DESIGN_LIMIT_MISMATCH"]


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_limit_check_negative_pressure_within_tolerance_matches():
    pid_limits = {"V-1": {"design_pressure_psig": -14.3, "design_temperature_f": "100"}}
    sop_limits = {"V-1": [EquipmentLimit("V-1", "V-1 vacuum", -14.7, "100")]}  # 0.4 apart

    assert cross_reference_limits(pid_limits, sop_limits) == []


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_limit_check_temperature_digit_concatenation_fixed():
    """Regression: _digits_only strips ALL non-digits, so a single value "125" used
    to match a semantically different range "1-25" (both -> "125"), silently missing
    a real mismatch. Found by independent review; fixed via _has_range_hyphen, which
    requires both sides to agree on "is this a range" before accepting a digit match —
    scoped narrowly to non-leading hyphens so it doesn't disturb the "/"-as-range-or-
    OCR-noise handling already validated against real data (see limit_check.py)."""
    pid_limits = {"E-1": {"design_pressure_psig": 300.0, "design_temperature_f": "125"}}
    sop_limits = {"E-1": [EquipmentLimit("E-1", "E-1", 300.0, "1-25")]}

    findings = cross_reference_limits(pid_limits, sop_limits)
    assert [f.category for f in findings] == ["DESIGN_LIMIT_MISMATCH"]


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_limit_check_partial_pid_dict_degrades_gracefully():
    """Regression: pid dict used to be indexed with [], not .get() — a title-block
    extraction that produced a partial dict (a plausible OCR outcome) raised
    KeyError and aborted the whole cross-reference. Found by independent review;
    fixed by reading both fields via .get() so a missing key degrades to "nothing
    to check against" for that one field, instead of crashing the whole pass."""
    pid_limits = {"F-715": {"design_temperature_f": "100"}}  # no design_pressure_psig key
    sop_limits = {"F-715": [EquipmentLimit("F-715", "F-715", 275.0, "100")]}

    assert cross_reference_limits(pid_limits, sop_limits) == []


# ---------------- component_types.type_from_tag ----------------

@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model  # same "unrecognized input -> None" invariant Sonnet's
# test_type_from_tag_unknown_prefix_edge_case covers independently
def test_type_from_tag_empty_and_separator_only_return_none():
    assert type_from_tag("") is None
    assert type_from_tag("   ") is None
    assert type_from_tag("-123") is None  # empty head before hyphen


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_type_from_tag_unusual_whitespace_casing_and_multi_hyphen():
    assert type_from_tag("  ft-101  ") == "flow_transmitter"     # padding + lowercase
    assert type_from_tag("PI 715A") == "pressure_indicator"       # space-separated bubble tag
    assert type_from_tag("PSV-501-A") == "safety_valve"           # multiple hyphens, split at first
    assert type_from_tag("FT101") == "flow_transmitter"           # no separator at all


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model
def test_type_from_tag_prefix_of_longer_prefix_not_confused():
    # "P" (pump) is a prefix of "PT" (pressure_transmitter); parser must use the
    # full head before the separator, not a shortest-prefix match.
    assert type_from_tag("PT-100") == "pressure_transmitter"
    assert type_from_tag("P-100") == "pump"
    assert type_from_tag("PSV-100") == "safety_valve"  # not "P"/pump, not "PS"


# ---------------- llm_ocr_assist.read_tag_with_llm ----------------

@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.failure_path
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_malformed_json_degrades(mock_post):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
    mock_post.return_value = resp

    assert read_tag_with_llm(_image(), _shape()) == (None, "")


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.failure_path
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_http_500_degrades(mock_post):
    import requests

    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    mock_post.return_value = resp

    assert read_tag_with_llm(_image(), _shape()) == (None, "")


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_multiple_tags_picks_first(mock_post):
    mock_post.return_value = _mock_response("Probably FT-101 but could be PT-202 or T-303.")
    tag, raw = read_tag_with_llm(_image(), _shape())
    assert tag == "FT-101"  # first LOOSE_TAG_PATTERN match wins


@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@patch("src.pid_extraction.llm_ocr_assist.requests.post")
def test_read_tag_with_llm_unknown_prefix_still_extracts_tag_and_normalizes_ws(mock_post):
    # Only an EXACT "UNKNOWN" short-circuits; "UNKNOWN ..." with a tag still parses,
    # and internal whitespace in a bubble tag is collapsed to a single space.
    mock_post.return_value = _mock_response("UNKNOWN, but maybe PI   715A")
    tag, raw = read_tag_with_llm(_image(), _shape())
    assert tag == "PI 715A"


# ---------------- tag_extractor.extract_sop_facts ----------------

@pytest.mark.unit
@pytest.mark.adversarial
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
def test_extract_sop_facts_chained_connection_drops_middle_pair():
    # "A connects to B connects to C": finditer is non-overlapping, so B is
    # consumed by the first match and the B->C connection is never captured.
    facts = extract_sop_facts("V-101 connects to P-101 connects to T-102.")
    assert ("P-101", "V-101") in facts.stated_connections
    assert ("P-101", "T-102") not in facts.stated_connections  # missed chained link


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_opus
@pytest.mark.multi_model
def test_extract_sop_facts_self_connection_is_dropped():
    facts = extract_sop_facts("Recirculation: P-101 connects to P-101.")
    assert facts.stated_connections == set()  # a==b guarded
    assert "P-101" in facts.referenced_tags
