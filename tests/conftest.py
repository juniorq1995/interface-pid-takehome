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
