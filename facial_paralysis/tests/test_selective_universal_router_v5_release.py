from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402


REPORT = ROOT / "docs/results/artifacts/universal_clinical_router_v5_candidate/report.json"
CANDIDATES = ROOT / "docs/model_candidates.json"
SUMMARY = ROOT / "docs/results/universal_clinical_router_v5_candidate.md"
V4_MODEL = ROOT / "docs/results/artifacts/universal_clinical_router_v4/model.json"
V4_REPORT = ROOT / "docs/results/artifacts/universal_clinical_router_v4/report.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rejected_candidate_is_registered_but_not_current(c: Check):
    registry = json.loads(CANDIDATES.read_bytes())
    c.eq(registry["schema_version"], "facial_paralysis_model_candidates_v1")
    c.eq(registry["current_model"], "universal_clinical_router_v4")
    matches = [
        row for row in registry["candidates"]
        if row["name"] == "universal_clinical_router_v5_candidate"
    ]
    c.eq(len(matches), 1)
    candidate = matches[0]
    c.eq(candidate["name"], "universal_clinical_router_v5_candidate")
    c.eq(candidate["status"], "rejected_not_promoted")
    c.true(not candidate["default_import"] and not candidate["promotion_authorized"])
    c.eq(candidate["report_sha256"], _sha(REPORT))


def test_machine_report_is_exact_reproducible_and_aggregate_only(c: Check):
    report = json.loads(REPORT.read_bytes())
    c.eq(_sha(REPORT), "97555485fdfc14253ffc6deb782a0f6ca2cf443a339d79bad588f18876d3c33a")
    c.true(not report["decision"]["passed"])
    c.eq(report["decision"]["selected"], None)
    c.eq(report["v4_model_sha256"], _sha(V4_MODEL))
    c.eq(report["v4_report_sha256"], _sha(V4_REPORT))
    lowered = REPORT.read_bytes().lower()
    for token in (
        b"anonymous_groups", b"labels", b"final_probability",
        b"component_probability", b"selection_sha256", b"/users/", b"/home/",
    ):
        c.true(token not in lowered, f"aggregate release excludes {token!r}")


def test_v4_surfaces_remain_the_only_default(c: Check):
    current = json.loads((ROOT / "docs/model_registry.json").read_bytes())
    c.eq(current["current"]["name"], "universal_clinical_router_v4")
    c.eq(_sha(V4_MODEL), "c8f8c217d508b15bf0d8626b42cead857192ecd738b1fffab94f364c6ed80495")
    c.eq(_sha(V4_REPORT), "56379e252fd6c88d74a98a89241bdbf4a96b84080f18a6055a41f880c8b34d8a")
    text = SUMMARY.read_text()
    c.true("候选不晋升" in text and "第四版继续作为当前主模型" in text)
    c.true("临床正确率" not in text and "临床准确率" not in text)


def test_primary_gate_metrics_explain_the_rejection(c: Check):
    report = json.loads(REPORT.read_bytes())
    margin = {
        profile: row["candidates"]["probability_margin"]["0.70"]
        for profile, row in report["evaluations"].items()
    }
    c.true(margin["free_asymmetry"]["balanced_accuracy"] >= 0.95)
    c.true(margin["scripted_multimechanism"]["accuracy"] >= 0.95)
    c.true(margin["scripted_multimechanism"]["balanced_accuracy"] < 0.95)
    c.true(margin["cue_aligned_upper"]["accuracy"] < 0.95)


if __name__ == "__main__":
    run_all("test_selective_universal_router_v5_release", dict(globals()))
