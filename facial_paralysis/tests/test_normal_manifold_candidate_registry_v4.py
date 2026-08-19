from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.normal_manifold_candidate_registry_v4 import (
    NORMAL_MANIFOLD_RATIONALE,
    NormalManifoldCandidateV4,
    candidate_registry_v4,
)


def test_registry_is_closed_unique_and_exact(c):
    candidates = candidate_registry_v4()
    c.eq(len(candidates), 6)
    c.eq(len({candidate.candidate_id for candidate in candidates}), 6)
    c.eq({candidate.normal_weight for candidate in candidates}, {0.0, 0.05, 0.2})
    c.eq({candidate.universal_blend for candidate in candidates}, {0.25, 0.5})
    c.true(all(type(candidate) is NormalManifoldCandidateV4 for candidate in candidates))
    c.eq(candidates[0].candidate_id, "NMR4-000")
    c.eq(candidates[-1].candidate_id, "NMR4-005")


def test_medical_rationale_has_limits_and_primary_evidence(c):
    c.eq(set(NORMAL_MANIFOLD_RATIONALE), {"healthy_anchor", "universal_normality"})
    for component in NORMAL_MANIFOLD_RATIONALE.values():
        c.eq(set(component), {"phenomenon", "evidence", "valid_labels", "contraindication"})
        c.true(len(component["evidence"]) >= 1)
        c.true(all(str(item).startswith("https://") for item in component["evidence"]))
        c.true(bool(component["contraindication"]))


def test_candidates_reject_non_frozen_values(c):
    forged = NormalManifoldCandidateV4("NMR4-999", 0.1, 0.5)
    c.true(forged not in candidate_registry_v4())


if __name__ == "__main__":
    run_all("test_normal_manifold_candidate_registry_v4", dict(globals()))
