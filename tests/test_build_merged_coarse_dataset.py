from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_merged_coarse_dataset import (
    DIGITIZE_PID_CLASS_MAP,
    EQUIPMENT,
    INSTRUMENT,
    VALVE,
    _relabel_file,
    _roboflow_class_map,
)


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_relabel_file_remaps_kept_classes_and_preserves_box_geometry(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("0 0.5 0.5 0.1 0.1\n14 0.2 0.3 0.05 0.05\n")
    dst = tmp_path / "dst.txt"

    box_count = _relabel_file(src, dst, DIGITIZE_PID_CLASS_MAP)

    assert box_count == 2
    lines = dst.read_text().strip().split("\n")
    assert lines[0] == f"{VALVE} 0.5 0.5 0.1 0.1"
    assert lines[1] == f"{INSTRUMENT} 0.2 0.3 0.05 0.05"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_relabel_file_drops_unmapped_ambiguous_classes_edge_case(tmp_path):
    """Classes 3/7/9/20/23 were deliberately excluded (ambiguous triangles,
    an unclear high-frequency class, a flow-direction arrow) rather than
    guessed into a bucket — confirm they're dropped, not defaulted to 0."""
    src = tmp_path / "src.txt"
    src.write_text("20 0.5 0.5 0.1 0.1\n3 0.1 0.1 0.02 0.02\n0 0.6 0.6 0.1 0.1\n")
    dst = tmp_path / "dst.txt"

    box_count = _relabel_file(src, dst, DIGITIZE_PID_CLASS_MAP)

    assert box_count == 1
    assert dst.read_text().strip() == f"{VALVE} 0.6 0.6 0.1 0.1"


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_relabel_file_all_classes_excluded_writes_empty_file_failure_path(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("20 0.5 0.5 0.1 0.1\n23 0.1 0.1 0.02 0.02\n")
    dst = tmp_path / "dst.txt"

    box_count = _relabel_file(src, dst, DIGITIZE_PID_CLASS_MAP)

    assert box_count == 0
    assert dst.read_text() == ""


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_roboflow_class_map_groups_valve_subtypes_to_single_coarse_class():
    names = ["Gate Valve", "Ball Valve", "Pressure Transmitter", "Tank", "Not Gate"]
    mapping = _roboflow_class_map(names)

    assert mapping[0] == VALVE
    assert mapping[1] == VALVE
    assert mapping[2] == INSTRUMENT
    assert mapping[3] == EQUIPMENT
    assert 4 not in mapping  # "Not Gate" doesn't match any coarse keyword


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_digitize_pid_class_map_covers_exactly_the_visually_confirmed_classes_edge_case():
    """Locks the exact set decided by visual inspection this session — a
    change here should be a deliberate re-labeling decision, not a silent
    drift picked up by refactoring."""
    assert set(DIGITIZE_PID_CLASS_MAP.keys()) == {
        0, 1, 2, 4, 5, 6, 8, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 24, 25, 26, 27, 28, 29, 30, 31,
        19, 21, 22,
    }
    excluded = {3, 7, 9, 20, 23}
    assert excluded.isdisjoint(DIGITIZE_PID_CLASS_MAP.keys())
