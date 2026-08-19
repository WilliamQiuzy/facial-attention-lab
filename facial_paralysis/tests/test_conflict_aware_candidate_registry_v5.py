from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.conflict_aware_candidate_registry_v5 import candidate_registry_v5


def test_registry_is_exact_four_and_locks_v4_representation(c):
    candidates = candidate_registry_v5()
    c.eq(len(candidates), 4)
    c.eq({item.projection_scope for item in candidates}, {"patient_block", "all_shared"})
    c.eq({item.projection_strength for item in candidates}, {0.5, 1.0})
    c.true(all(item.base_candidate_id == "NMR4-001" for item in candidates))
    c.eq(len({item.candidate_id for item in candidates}), 4)


if __name__ == "__main__":
    run_all("test_conflict_aware_candidate_registry_v5", dict(globals()))
