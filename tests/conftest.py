import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from scripts.generate_sample_data import PID_PATH, SOP_PATH, generate_pid, generate_sop


@pytest.fixture(scope="session", autouse=True)
def sample_data():
    if not PID_PATH.exists():
        generate_pid()
    if not SOP_PATH.exists():
        generate_sop()


# --- Qbital test quality score (type/depth/source marks) ---
# See memory: feedback-test-quality-standard. Target: A (>=85/100).
TYPE_MARKS = {"unit", "integration", "e2e", "pipeline", "smoke", "regression", "property", "adversarial"}
DEPTH_MARKS = {"happy_path", "edge_case", "failure_path", "concurrent"}
SOURCE_MARKS = {
    "authored_claude_opus",
    "authored_claude_sonnet",
    "authored_claude_haiku",
    "authored_gpt4",
    "authored_gemini_pro",
    "authored_human",
}

_collected_items = []


def pytest_collection_modifyitems(items):
    _collected_items[:] = items


def pytest_terminal_summary(terminalreporter):
    total = len(_collected_items)
    if total == 0:
        return

    types_seen, depths_seen, sources_seen = set(), set(), set()
    multi_model_count = 0
    unmarked_count = 0

    for item in _collected_items:
        marks = {m.name for m in item.iter_markers()}
        item_types = marks & TYPE_MARKS
        item_depths = marks & DEPTH_MARKS
        item_sources = marks & SOURCE_MARKS
        types_seen |= item_types
        depths_seen |= item_depths
        sources_seen |= item_sources
        if "multi_model" in marks:
            multi_model_count += 1
        if not item_types and not item_depths and not item_sources:
            unmarked_count += 1

    type_score = max(0, 30 - 4 * len(TYPE_MARKS - types_seen))
    depth_score = max(0, 25 - 6 * len(DEPTH_MARKS - depths_seen))
    source_score = min(25, 5 * len(sources_seen))
    multi_model_score = min(20, 2 * multi_model_count)
    unmarked_penalty = 20 * (unmarked_count / total)
    raw_total = type_score + depth_score + source_score + multi_model_score - unmarked_penalty
    final_score = round(max(0, min(100, raw_total)), 1)

    grade = "A" if final_score >= 85 else "B" if final_score >= 70 else "C" if final_score >= 55 else "D"

    terminalreporter.write_sep("=", "Qbital Test Quality Score")
    terminalreporter.write_line(f"Tests collected:     {total}")
    terminalreporter.write_line(f"Types covered:        {len(types_seen)}/{len(TYPE_MARKS)}  -> {type_score}/30")
    terminalreporter.write_line(f"Depths covered:       {len(depths_seen)}/{len(DEPTH_MARKS)}  -> {depth_score}/25")
    terminalreporter.write_line(f"Distinct sources:     {len(sources_seen)}/{len(SOURCE_MARKS)}  -> {source_score}/25")
    terminalreporter.write_line(f"Multi-model tests:    {multi_model_count}  -> {multi_model_score}/20")
    terminalreporter.write_line(f"Unmarked tests:       {unmarked_count}/{total}  -> -{round(unmarked_penalty, 1)}")
    terminalreporter.write_line(f"SCORE: {final_score}/100 ({grade})  [target: A, >=85]")
    if types_seen != TYPE_MARKS:
        terminalreporter.write_line(f"  missing types:  {sorted(TYPE_MARKS - types_seen)}")
    if depths_seen != DEPTH_MARKS:
        terminalreporter.write_line(f"  missing depths: {sorted(DEPTH_MARKS - depths_seen)}")
