from pathlib import Path

import pytest

from src.sop_extraction.docx_parser import parse_sop
from src.sop_extraction.tag_extractor import extract_sop_facts


def test_extract_sop_facts_happy_path():
    text = "Open valve V-101 before starting pump P-101. V-101 connects to P-101."
    facts = extract_sop_facts(text)

    assert facts.referenced_tags == {"V-101", "P-101"}
    assert facts.declared_types["V-101"] == "valve"
    assert facts.declared_types["P-101"] == "pump"
    assert ("P-101", "V-101") in facts.stated_connections


def test_extract_sop_facts_tag_without_type_keyword_edge_case():
    text = "Check T-101 during the morning walkdown."
    facts = extract_sop_facts(text)

    assert facts.referenced_tags == {"T-101"}
    assert "T-101" not in facts.declared_types
    assert facts.stated_connections == set()


def test_extract_sop_facts_empty_text_failure_case():
    facts = extract_sop_facts("")
    assert facts.referenced_tags == set()
    assert facts.declared_types == {}
    assert facts.stated_connections == set()


def test_parse_sop_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_sop(tmp_path / "does_not_exist.docx")
