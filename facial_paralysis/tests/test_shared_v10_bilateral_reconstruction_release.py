from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/results/artifacts/shared_v10_bilateral_reconstruction/report.json"
REPORT_SHA256 = "e13cd5a5d72e77c94f3181a90a86d7e84f9fe6238d41cc2c5099589e3159157a"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_aggregate_release_is_complete_and_nonpromoting(c):
    c.eq(_sha(ARTIFACT), REPORT_SHA256)
    report = json.loads(ARTIFACT.read_text())
    c.eq(report["schema_version"], "bilateral_reconstruction_shared_v10_search")
    c.eq(report["counts"], {"palsynet": 38, "neuroface": 36, "meei": 56})
    c.eq(len(report["candidate_registry"]), 6)
    c.eq(len(report["evaluations"]), 6)
    c.eq(report["selection"]["seeds"], [0, 1, 2])
    c.eq(report["decision"], {
        "clinical_claim_authorized": False,
        "deployment_model_changed": False,
        "promoted_research_candidate_id": None,
        "research_promotion_authorized": False,
    })
    c.true(not any(
        row["promotion_gate_passed"] for row in report["summaries"].values()
    ))
    c.eq(report["audit"], {
        "mayo_predictions": 0,
        "mayo_reads": 0,
        "palsynet_protected_reads": 0,
    })
    emitted = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "probabilities", "group_id", "participant_id", "patient_id",
        "recording_id", "/users/", "/home/",
    ):
        c.true(forbidden not in emitted)


def test_current_research_model_is_blv9_009_while_deployment_stays_v8(c):
    current = (ROOT / "docs/CURRENT_RESEARCH_MODEL.md").read_text()
    c.true("BLV9-009" in current)
    c.true("Masked Clinical Reconstruction" in current)
    c.true("BRV10-000" in current)
    c.true("V8 remains the deployment model" in current)
    c.true("Mayo performance is unknown" in current)
    c.eq(
        _sha(ROOT / "docs/archive/models/model_registry_v2_ucr4.json"),
        "67ce23f2fb3155e181d5615c69e721ec19e253bb7797426587ed4bef5e63f489",
    )
    c.eq(
        _sha(ROOT / "releases/shared-v8-deployment-v1/model_manifest.json"),
        "d40721a8fbda0c37a7e30a49f68bc94aa943e34c35e21864a1e79a2597676cbc",
    )


def test_public_result_explains_why_v10_did_not_replace_v9(c):
    text = (ROOT / "docs/results/shared_v10_bilateral_reconstruction.md").read_text()
    for required in (
        "BLV9-009 remains the research baseline",
        "162 fits",
        "bilateral decomposition",
        "unordered twin",
        "minimum AUROC",
        "Mayo reads: **0**",
        "V8 remains the deployment model",
    ):
        c.true(required in text)
    c.true("Mayo accuracy" not in text)


if __name__ == "__main__":
    run_all("test_shared_v10_bilateral_reconstruction_release", dict(globals()))
