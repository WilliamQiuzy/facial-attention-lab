from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/results/artifacts/broad_literature_shared_v9/report.json"
REPORT_SHA256 = "27762fafb4923f043483bfc481d70948b3aaff12141f0a5fe2dbfeb1756c7ac4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_machine_report_is_exact_complete_and_aggregate_only(c):
    c.eq(_sha(ARTIFACT), REPORT_SHA256)
    report = json.loads(ARTIFACT.read_text())
    c.eq(report["schema_version"], "broad_literature_shared_v9_search")
    c.eq(report["counts"], {"palsynet": 38, "neuroface": 36, "meei": 56})
    c.eq(report["selection"]["evaluated_new_models"], 20)
    c.eq(report["selection"]["seeds"], [0, 1, 2])
    c.eq(len(report["candidate_registry"]), 21)
    c.eq(len(report["evaluations"]), 21)
    c.eq(len(report["summaries"]), 21)
    c.eq(set(report["selection"]["ranking"]), set(report["evaluations"]))
    for evaluation in report["evaluations"].values():
        c.eq(set(evaluation), {"0", "1", "2"})
        c.true(all(seed["model_fits"] == 9 for seed in evaluation.values()))
    emitted = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "probabilities", "group_id", "participant_id", "patient_id",
        "recording_id", "/users/", "/home/",
    ):
        c.true(forbidden not in emitted)


def test_no_candidate_is_promoted_and_v8_remains_frozen(c):
    report = json.loads(ARTIFACT.read_text())
    c.eq(report["decision"], {
        "clinical_claim_authorized": False,
        "promoted_candidate_id": None,
        "promotion_authorized": False,
    })
    c.true(not any(
        summary["promotion_gate_passed"]
        for summary in report["summaries"].values()
    ))
    c.eq(report["audit"], {
        "mayo_predictions": 0,
        "mayo_reads": 0,
        "palsynet_protected_reads": 0,
    })
    c.eq(
        _sha(ROOT / "docs/model_registry.json"),
        "67ce23f2fb3155e181d5615c69e721ec19e253bb7797426587ed4bef5e63f489",
    )
    c.eq(
        _sha(ROOT / "docs/CURRENT_DEPLOYMENT_MODEL.md"),
        "702e3da45e1cdd19a04526046635442a5394c8dac0abf4baaa4d81381f342bf4",
    )


def test_public_report_names_all_families_and_preserves_claim_boundary(c):
    text = (ROOT / "docs/results/broad_literature_shared_v9.md").read_text()
    for required in (
        "20 mechanism-distinct shared",
        "SAM, ASAM, SWA, and R-Drop",
        "modality dropout and action-drop consistency",
        "VICReg, Barlow Twins",
        "focal, LDAM, pairwise AUROC",
        "progressive layered extraction, Cross-Stitch",
        "567 fits",
        "V8 remains the canonical deployment model",
        "Mayo reads, and Mayo predictions",
        "Mayo performance remains unknown",
    ):
        c.true(required in text)
    c.true("Mayo accuracy" not in text)


if __name__ == "__main__":
    run_all("test_broad_literature_shared_v9_release", dict(globals()))
