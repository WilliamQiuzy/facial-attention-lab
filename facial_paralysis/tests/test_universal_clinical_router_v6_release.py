from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402
from scripts.run_dense_action_router_v6 import _implementation_sha256  # noqa: E402


REPORT = ROOT / "docs/results/artifacts/universal_clinical_router_v6_candidate/report.json"
CANDIDATES = ROOT / "docs/model_candidates.json"
SUMMARY = ROOT / "docs/results/universal_clinical_router_v6_candidate.md"
V4_MODEL = ROOT / "docs/results/artifacts/universal_clinical_router_v4/model.json"
V4_REPORT = ROOT / "docs/results/artifacts/universal_clinical_router_v4/report.json"
CURRENT_IMPORT = ROOT / "src/models/current.py"
V4_RUNTIME = ROOT / "src/models/universal_clinical_router_v4.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v6_is_registered_as_noncurrent_exposed_candidate(c: Check):
    registry = json.loads(CANDIDATES.read_bytes())
    c.eq(registry["current_model"], "universal_clinical_router_v4")
    candidates = {row["name"]: row for row in registry["candidates"]}
    c.eq(set(candidates), {
        "universal_clinical_router_v5_candidate",
        "universal_clinical_router_v6_candidate",
    })
    candidate = candidates["universal_clinical_router_v6_candidate"]
    c.eq(candidate["status"], "development_gate_passed_awaiting_untouched_validation")
    c.true(not candidate["default_import"] and not candidate["promotion_authorized"])
    c.eq(candidate["report_sha256"], _sha(REPORT))


def test_v6_exact_report_passes_all_three_locked_development_gates(c: Check):
    report = json.loads(REPORT.read_bytes())
    c.eq(_sha(REPORT), "f1f4368266db238b79bdd738baf68aeed7a1aff281f19f3f589fea942297b956")
    expected = {
        "palsynet_development": (38, 36 / 38, 0.9523809523809523),
        "neuroface": (36, 34 / 36, 0.96),
        "meei": (56, 53 / 56, 0.9673913043478262),
    }
    for profile, (participants, accuracy, balanced_accuracy) in expected.items():
        row = report["evaluations"][profile]
        c.eq(row["participants"], participants)
        c.true(abs(row["accuracy"] - accuracy) < 1e-12)
        c.true(abs(row["balanced_accuracy"] - balanced_accuracy) < 1e-12)
        c.true(row["accuracy"] >= 0.93 and row["balanced_accuracy"] >= 0.90)
    c.true(report["decision"]["all_profile_gate_passed"])
    c.true(not report["decision"]["promotion_authorized"])
    c.true(report["claim_boundary"]["candidate_configuration_selected_after_development_exploration"])
    c.true(not report["claim_boundary"]["untouched_external_validation"])


def test_report_binds_implementation_and_excludes_private_rows(c: Check):
    report = json.loads(REPORT.read_bytes())
    c.eq(report["audit"]["implementation_sha256"], _implementation_sha256(ROOT))
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    c.eq(report["audit"]["mayo_reads"], 0)
    lowered = REPORT.read_bytes().lower()
    for token in (
        b"anonymous_groups", b"labels", b"final_probability",
        b"component_probability", b"recording_id", b"participant_id",
        b"source_sha256", b"/users/", b"/home/",
    ):
        c.true(token not in lowered, f"aggregate release excludes {token!r}")


def test_ucr4_remains_byte_exact_only_default(c: Check):
    current = json.loads((ROOT / "docs/model_registry.json").read_bytes())
    c.eq(current["current"]["name"], "universal_clinical_router_v4")
    c.eq(_sha(V4_MODEL), "c8f8c217d508b15bf0d8626b42cead857192ecd738b1fffab94f364c6ed80495")
    c.eq(_sha(V4_REPORT), "56379e252fd6c88d74a98a89241bdbf4a96b84080f18a6055a41f880c8b34d8a")
    c.eq(_sha(CURRENT_IMPORT), "506f7b97c948c4f9e919981a6a771d9c03ee0194752f95f5912e65552e1fc8c4")
    c.eq(_sha(V4_RUNTIME), "c32825843d0f03af4ba08294ef17420b5f2816cf7e0dfd80daa4809bdf4d6d71")
    text = SUMMARY.read_text()
    c.true("第四版仍是唯一当前主模型" in text)
    c.true("尚未获准晋升" in text and "不能证明临床泛化" in text)


if __name__ == "__main__":
    run_all("test_universal_clinical_router_v6_release", dict(globals()))
