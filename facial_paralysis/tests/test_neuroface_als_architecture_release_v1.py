"""Public-release contracts for the NeuroFace ALS architecture experiment."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402


REPORT_PATH = (
    ROOT / "docs/results/artifacts/neuroface_als_architecture_v1/report.json"
)
SUMMARY_PATH = ROOT / "docs/results/neuroface_als_architecture_v1.md"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_release_decision_distinguishes_auroc_from_accuracy(c: Check):
    report = _report()
    decision = report["decision"]
    strict = report["strict_best"]
    c.eq(report["schema_version"], "neuroface_als_architecture_release_v1",
         "release schema is frozen")
    c.true(strict["auroc"] > 0.90 and strict["accuracy"] < 0.90,
           "strict evidence clears only the AUROC target")
    c.eq(decision["strict_auroc_above_0_90"], True,
         "AUROC decision agrees with the metric")
    c.eq(decision["strict_accuracy_above_0_90"], False,
         "accuracy decision agrees with the metric")
    c.eq(decision["beats_published_0_91_accuracy"], False,
         "release does not claim published accuracy superiority")
    c.eq(decision["beats_published_0_97_auroc"], False,
         "release does not claim published AUROC superiority")
    c.eq(decision["frozen_bells_palsy_110d_replaced"], False,
         "NeuroFace work cannot replace the Bell's-palsy model")


def test_release_counts_metrics_and_bootstrap_are_exact(c: Check):
    report = _report()
    endpoint = report["neuroface_endpoint"]
    strict = report["strict_best"]
    c.eq((endpoint["participants"], endpoint["positive_participants"],
          endpoint["negative_participants"], endpoint["recordings"]),
         (22, 11, 11, 66), "ALS-versus-control cohort is exact")
    c.eq(endpoint["tasks"], {
        "NSM_KISS": 22, "NSM_OPEN": 22, "NSM_SPREAD": 22,
    }, "each participant contributes the three frozen tasks")
    c.eq(strict["protocol"], "nested_participant_loso",
         "strict result is participant-disjoint and nested")
    c.eq(strict["accuracy"], 19 / 22,
         "strict accuracy is the exact participant-level fraction")
    c.eq(strict["sensitivity"], 9 / 11,
         "strict sensitivity is the exact ALS fraction")
    c.eq(strict["specificity"], 10 / 11,
         "strict specificity is the exact control fraction")
    c.eq(strict["bootstrap"]["replicates"], 5000,
         "participant bootstrap count is frozen")
    c.eq(strict["bootstrap"]["seed"], 20260814,
         "participant bootstrap seed is frozen")
    broader = report["broader_robustness_context"]
    c.eq((broader["participants"], broader["auroc"],
          broader["balanced_accuracy"]), (36, 0.753, 0.744),
         "separate cross-disease robustness evidence remains visible")
    c.true("not_pooled" in broader["interpretation"],
           "ALS and cross-disease endpoints cannot be pooled")


def test_release_contains_only_aggregate_public_evidence(c: Check):
    report = _report()
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    combined = report_text + "\n" + summary_text
    lower = combined.lower()
    for forbidden in (
        "grp_", "rec_", "/users/", "oof_probabilities", "participant_ids",
        "recording_ids", "raw_probabilities", "mayo_predictions",
    ):
        c.true(forbidden not in lower,
               f"public release excludes sensitive field {forbidden}")
    c.true("not in strict accuracy" in lower,
           "summary states that strict accuracy did not exceed 0.90")
    c.true("does not exceed the published" in lower,
           "summary explicitly rejects published-superiority wording")
    c.true("internal_development_only" in report_text,
           "machine report carries the development-only boundary")
    c.eq(report["claim_boundary"]["external_validation"], False,
         "release cannot be read as external validation")
    c.eq(report["claim_boundary"]["clinical_deployment_claim"], False,
         "release cannot be read as a deployment claim")


def test_provenance_is_complete_and_hash_shaped(c: Check):
    provenance = _report()["provenance"]
    expected = {
        "private_manifest_sha256", "dynamic_collection_sha256",
        "au_collection_sha256", "spread_au_cache_set_sha256",
        "environment_implementation_sha256",
        "final_spread_implementation_sha256",
        "multitask_implementation_sha256", "tcn_implementation_sha256",
        "final_spread_report_sha256", "multitask_report_sha256",
        "tcn_report_sha256",
    }
    c.eq(set(provenance), expected, "aggregate provenance fields are closed")
    c.true(all(re.fullmatch(r"[0-9a-f]{64}", value)
               for value in provenance.values()),
           "all evidence commitments are SHA-256 values")


if __name__ == "__main__":
    run_all("test_neuroface_als_architecture_release_v1", dict(globals()))
