"""Release contract for the research-only scripted action-capacity branch."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402


CURRENT = ROOT / "docs/results/current_development_model.json"
SUMMARY = ROOT / "docs/results/script_conditioned_action_capacity_v1.md"
NEUROFACE = (
    ROOT / "outputs/dynamic_landmark/benchmarks/external/"
    "neuroface-action-capacity-v1/report.json"
)
MAYO = (
    ROOT / "outputs/dynamic_landmark/benchmarks/external/"
    "mayo-action-anchor-feasibility-v1/report.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_model_keeps_110d_locked_and_records_research_branch(c: Check):
    current = json.loads(CURRENT.read_text())
    frozen = current["candidates"]["landmark_mi_110d"]
    c.eq(current["schema_version"], "facial_paralysis_current_model_v7")
    c.eq(current["model"]["name"], "landmark_mi_110d")
    c.eq(frozen["feature_dimension"], 110)
    c.eq(frozen["auroc"], 0.980392156862745)
    c.eq(frozen["balanced_accuracy"], 0.9523809523809523)
    c.eq(current["decision"]["locked_candidate"], "landmark_mi_110d")

    branch = current["research_only_branches"]["scripted_action_capacity_v1"]
    c.eq(branch["current_110d_replaced"], False)
    c.eq(branch["feature_dimension_per_expert"], 18)
    c.eq(branch["action_experts"], 3)
    c.eq(branch["neuroface_metrics"]["auroc"], 0.7527272727272728)
    c.eq(branch["neuroface_metrics"]["auroc_ci95"], [
        0.5781818181818182, 0.9018181818181817,
    ])
    c.eq(branch["neuroface_endpoint"], "als_or_post_stroke_vs_healthy_control")
    c.eq(branch["bell_palsy_transfer_claim_authorized"], False)
    c.eq(branch["fusion_authorized"], False)
    c.eq(branch["clinical_use_authorized"], False)


def test_reports_and_summary_are_exactly_bound(c: Check):
    current = json.loads(CURRENT.read_text())
    branch = current["research_only_branches"]["scripted_action_capacity_v1"]
    neuroface = json.loads(NEUROFACE.read_text())
    mayo = json.loads(MAYO.read_text())
    c.eq(branch["neuroface_public_report_sha256"], _sha256(NEUROFACE))
    c.eq(branch["mayo_timing_audit_public_report_sha256"], _sha256(MAYO))
    c.eq(neuroface["decision"]["current_110d_replaced"], False)
    c.eq(neuroface["decision"]["fusion_authorized"], False)
    c.eq(mayo["media_inventory"]["source_files"], 53)
    c.eq(mayo["media_inventory"]["unique_contents"], 52)
    c.eq(mayo["media_inventory"]["audio_bearing_source_files"], 51)
    c.eq(mayo["media_inventory"]["audio_free_source_files"], 2)
    c.eq(mayo["timing_gate"]["eligible"], False)
    c.eq(mayo["scoring"]["mayo_action_expert_predictions"], 0)

    summary = SUMMARY.read_text()
    normalized_summary = " ".join(summary.split())
    c.true("0.753" in summary and "0.578–0.902" in summary)
    c.true("53" in summary and "52" in summary)
    c.true("Fifty-one source files contain audio" in summary)
    c.true("frozen 110D" in summary)
    c.true("not Bell's palsy" in summary)
    c.true(
        "not a causal or head-to-head representation comparison"
        in normalized_summary
    )
    c.true("participant-disjoint" in normalized_summary)
    c.true("reported separately before any fusion" in normalized_summary)
    c.true("Fusion must then be preregistered" in normalized_summary)
    c.true("untouched split or cohort" in normalized_summary)

    neuroface_summary = (
        ROOT / "docs/results/neuroface_action_capacity_v1.md"
    ).read_text()
    normalized_neuroface_summary = " ".join(neuroface_summary.split())
    c.true(
        "not a causal or head-to-head representation comparison"
        in normalized_neuroface_summary
    )
    c.true(
        "reported separately before any fusion" in normalized_neuroface_summary
    )


if __name__ == "__main__":
    run_all("test_script_conditioned_action_capacity_release_v1", dict(globals()))
