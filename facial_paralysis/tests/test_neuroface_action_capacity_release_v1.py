"""Published aggregate release contract for NeuroFace capacity v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _testlib import Check, run_all


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "outputs" / "dynamic_landmark" / "benchmarks" / "external"
    / "neuroface-action-capacity-v1" / "report.json"
)
SUMMARY = ROOT / "docs" / "results" / "neuroface_action_capacity_v1.md"
REPORT_SHA256 = "be246d848e78598c47d0470ff5c4175dbe1a4b6b012fadedbbf0f7494f34c290"


def test_public_report_is_exact_aggregate_only(c: Check):
    payload = REPORT.read_bytes()
    c.eq(hashlib.sha256(payload).hexdigest(), REPORT_SHA256)
    report = json.loads(payload)
    c.eq(report["schema_version"], "neuroface_action_capacity_v1")
    c.eq(report["dataset"]["participants"], 36)
    c.eq(report["dataset"]["primary_task_recordings"], 108)
    c.eq(report["bootstrap"]["valid_draws"], 5000)
    c.eq(report["bootstrap"]["invalid_draws"], 0)
    c.eq(report["metrics"]["auroc"]["point"], 0.7527272727272728)
    c.eq(report["metrics"]["auroc"]["ci95"]["lower"], 0.5781818181818182)
    c.eq(report["audit"], {
        "palsynet_cache_reads": 0,
        "palsynet_path_accesses": 0,
        "palsynet_predictions": 0,
    })
    c.eq(report["decision"]["capacity_feasibility_signal"], True)
    c.eq(report["decision"]["current_110d_replaced"], False)
    text = payload.decode("utf-8")
    for forbidden in ("grp_", "/home/", "participant_id", "recording_id"):
        c.true(forbidden not in text, f"public aggregate excludes {forbidden}")


def test_human_summary_binds_the_machine_report_and_claim_boundary(c: Check):
    summary = SUMMARY.read_text(encoding="utf-8")
    c.true(REPORT_SHA256 in summary)
    c.true("AUROC | 0.753 | 0.578–0.902" in summary)
    c.true("does **not** authorize" in summary)
    c.true("It is not Bell's palsy detection" in summary)


if __name__ == "__main__":
    run_all("test_neuroface_action_capacity_release_v1", dict(globals()))
