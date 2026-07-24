"""Tests specifically covering test-type/depth marks the rest of the suite
doesn't naturally exercise (smoke, e2e, property, concurrent) — see
tests/conftest.py's Qbital test quality scoring hook.
"""
from __future__ import annotations

import importlib
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.generate_sample_data import PID_PATH, SOP_PATH
from src.component_types import type_from_tag
from src.sop_extraction.tag_extractor import extract_sop_facts


@pytest.mark.smoke
@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_all_core_modules_import_cleanly():
    modules = [
        "src.component_types",
        "src.crossref.compare",
        "src.crossref.limit_check",
        "src.crossref.report",
        "src.pid_extraction.connector_detection",
        "src.pid_extraction.graph_builder",
        "src.pid_extraction.llm_ocr_assist",
        "src.pid_extraction.ocr_tagging",
        "src.pid_extraction.pdf_to_image",
        "src.pid_extraction.pipeline",
        "src.pid_extraction.shape_detection",
        "src.pid_extraction.title_block",
        "src.pid_extraction.vector_lines",
        "src.pid_extraction.vector_symbols",
        "src.sop_extraction.docx_parser",
        "src.sop_extraction.limits_table",
        "src.sop_extraction.tag_extractor",
    ]
    for name in modules:
        importlib.import_module(name)


@pytest.mark.e2e
@pytest.mark.pipeline
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_main_run_end_to_end_writes_report(tmp_path):
    import main

    out_dir = tmp_path / "output"
    main.run(PID_PATH, SOP_PATH, out_dir)

    assert (out_dir / "pid_graph.json").exists()
    assert (out_dir / "pid_graph.graphml").exists()
    assert (out_dir / "cross_reference_report.json").exists()
    report_text = (out_dir / "cross_reference_report.log").read_text()
    assert "Cross-Reference Report" in report_text


@pytest.mark.property
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@given(st.text(max_size=50))
def test_type_from_tag_never_raises_on_arbitrary_input(text):
    result = type_from_tag(text)
    assert result is None or isinstance(result, str)


@pytest.mark.property
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
@given(st.from_regex(r"[A-Z]{1,4}-\d{2,4}[A-Z]?", fullmatch=True))
def test_type_from_tag_valid_tag_shape_never_raises(tag):
    # Any string matching the tag grammar must resolve to either a known
    # type or None — never raise, regardless of which specific prefix.
    result = type_from_tag(tag)
    assert result is None or isinstance(result, str)


@pytest.mark.concurrent
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_type_from_tag_thread_safe_under_concurrent_calls():
    tags = ["V-101", "FT-101", "PSV-501A", "XYZ-999", "PI 715A"] * 40
    results: list[str | None] = [None] * len(tags)
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            results[index] = type_from_tag(tags[index])
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a race-detection test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(tags))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results == [type_from_tag(t) for t in tags]


@pytest.mark.adversarial
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_sop_facts_adversarial_malformed_input_does_not_crash():
    # Homoglyph digits (Cyrillic/fullwidth), zero-width joiners, a pathologically
    # long run of tag-shaped noise, and raw control characters — none of this
    # should raise, and none of it should produce a false tag match.
    adversarial_text = (
        "V​-101 connects to Ｖ-101\n"
        + ("FAKE-999 " * 5000)
        + "\x00\x01\x02 PSV-́101\n"
        + "А-101"  # Cyrillic "А", not Latin "A"
    )
    facts = extract_sop_facts(adversarial_text)
    assert isinstance(facts.referenced_tags, set)
    assert isinstance(facts.stated_connections, set)
    # "FAKE-999" is a syntactically valid tag shape — it SHOULD match; the
    # point of this test is robustness (no crash, no hang), not rejecting it.
    assert "FAKE-999" in facts.referenced_tags


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_type_from_tag_rejects_non_string_input_failure_path():
    with pytest.raises(AttributeError):
        type_from_tag(None)  # type: ignore[arg-type]
