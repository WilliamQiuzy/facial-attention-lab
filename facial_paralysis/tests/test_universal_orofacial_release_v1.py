"""Public release invariants for Universal Orofacial research v1/v2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402

REPORT = ROOT / "docs/results/artifacts/universal_orofacial_v1/report.json"
SUMMARY = ROOT / "docs/results/universal_orofacial_v1.md"


def test_public_release_records_negative_gate_without_overwriting_models(c: Check):
    report = json.loads(REPORT.read_text())
    c.eq(report["schema_version"], "universal_orofacial_research_summary_v1",
         "aggregate release schema is frozen")
    c.eq(report["counts"], {
        "development_participants": 74,
        "neuroface_participants": 36,
        "palsynet_development_participants": 38,
        "meei_diagnostic_participants": 60,
    }, "all metric units are participants")
    c.true(not report["decision"]["universal_model_promoted"],
           "failed cross-source evidence cannot replace endpoint models")
    c.eq(report["multisignal_v2"]["locked_representation"], "fusion_398",
         "best representation is retained as a research artifact only")
    c.eq(report["multisignal_v2"]["metrics"]["neuroface"]["auroc"], 0.68,
         "NeuroFace shortfall is public, not hidden")
    c.eq(report["meei_locked_diagnostic"]["metrics"]["auroc"], 0.82,
         "MEEI uses the locked v1 candidate without refit")
    encoded = json.dumps(report, sort_keys=True)
    c.true(all(token not in encoded for token in (
        "grp_", "rec_", "/Users/", "probabilities", "coefficient",
    )), "aggregate release contains no private identifiers or model weights")


def test_summary_distinguishes_universal_failure_from_endpoint_results(c: Check):
    text = SUMMARY.read_text()
    for required in (
        "not promoted", "AUROC", "balanced accuracy", "PalsyNet protected",
        "endpoint-specific", "not clinical validation", "Py-Feat",
    ):
        c.true(required in text, f"summary states {required!r}")


if __name__ == "__main__":
    run_all("test_universal_orofacial_release_v1", dict(globals()))
