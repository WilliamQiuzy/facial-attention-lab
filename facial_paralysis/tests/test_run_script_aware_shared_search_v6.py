from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from scripts import run_script_aware_shared_search_v6 as runner
from src.models.script_aware_shared_router_v6 import candidate_registry_v6


def test_phase_and_cli_boundary(c):
    ids = tuple(item.candidate_id for item in candidate_registry_v6())
    runner.validate_candidate_phase("screen", ids)
    runner.validate_candidate_phase("confirm", ids[:2])
    c.raises(lambda: runner.validate_candidate_phase("screen", ids[:-1]), ValueError)
    source = inspect.getsource(runner._parser).lower()
    c.true("mayo" not in source and "outer-test" not in source)


if __name__ == "__main__":
    run_all("test_run_script_aware_shared_search_v6", dict(globals()))
