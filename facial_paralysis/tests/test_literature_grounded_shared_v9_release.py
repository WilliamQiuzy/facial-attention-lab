from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/results/artifacts/literature_grounded_shared_v9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_artifacts_are_aggregate_complete_and_nonpromoting(c):
    diagnostic = json.loads((ARTIFACTS / "optimization_diagnostic.json").read_text())
    literature = json.loads((ARTIFACTS / "literature_screen.json").read_text())
    experts = json.loads(
        (ARTIFACTS / "multigate_shared_expert_screen.json").read_text()
    )
    c.true(not diagnostic["cagrad_authorized"])
    c.true(not diagnostic["gradnorm_authorized"])
    for report in (literature, experts):
        c.eq(report["counts"], {"palsynet": 38, "neuroface": 36, "meei": 56})
        c.eq(report["audit"], {
            "palsynet_protected_reads": 0,
            "mayo_reads": 0,
            "mayo_predictions": 0,
        })
        c.true(not report["decision"]["promotion_authorized"])
        c.true(not report["decision"]["clinical_claim_authorized"])
        emitted = json.dumps(report, sort_keys=True).lower()
        c.true("probabilities" not in emitted and "group_id" not in emitted)
        c.true("/users/" not in emitted and "patient_id" not in emitted)


def test_v8_registry_and_deployment_remain_byte_frozen(c):
    c.eq(
        _sha(ROOT / "docs/model_registry.json"),
        "67ce23f2fb3155e181d5615c69e721ec19e253bb7797426587ed4bef5e63f489",
    )
    c.eq(
        _sha(ROOT / "docs/CURRENT_DEPLOYMENT_MODEL.md"),
        "702e3da45e1cdd19a04526046635442a5394c8dac0abf4baaa4d81381f342bf4",
    )


def test_public_decision_distinguishes_experiments_from_rejected_ideas(c):
    text = (ROOT / "docs/results/literature_grounded_shared_v9.md").read_text()
    for required in (
        "Bilateral anatomical relation residual",
        "Clinical-kinematic auxiliary supervision",
        "Multi-gate shared experts",
        "Not counted as experiments",
        "V8 remains",
        "Mayo reads: **0**",
    ):
        c.true(required in text)
    c.true("Mayo accuracy" not in text)


if __name__ == "__main__":
    run_all("test_literature_grounded_shared_v9_release", dict(globals()))
