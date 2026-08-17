from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from scripts.run_dense_action_router_v6 import (
    EXPECTED_RUNTIME_VERSIONS,
    LOCKED_DECISION_THRESHOLD,
    PROFILE_REGISTRY,
    _json_document,
    _read_exact_bytes,
    build_public_report,
    write_public_report_no_overwrite,
)


def _metrics():
    return {
        "palsynet_development": {
            "participants": 38, "accuracy": 36 / 38,
            "balanced_accuracy": 0.9523809523809523,
            "sensitivity": 0.9047619047619048,
            "specificity": 1.0, "auroc": 0.9803921568627451,
            "brier": 0.08, "errors": 2,
        },
        "neuroface": {
            "participants": 36, "accuracy": 34 / 36,
            "balanced_accuracy": 0.96, "sensitivity": 0.92,
            "specificity": 1.0, "auroc": 0.989090909090909,
            "brier": 0.07, "errors": 2,
        },
        "meei": {
            "participants": 56, "accuracy": 53 / 56,
            "balanced_accuracy": 0.9673913043478262,
            "sensitivity": 43 / 46, "specificity": 1.0,
            "auroc": 0.9456521739130436, "brier": 0.09,
            "errors": 3,
        },
    }


def _provenance():
    return {
        "ucr4_artifact_sha256": "1" * 64,
        "neuroface_collection_sha256": "2" * 64,
        "meei_collection_sha256": "3" * 64,
        "implementation_sha256": "4" * 64,
        "runtime_versions": dict(EXPECTED_RUNTIME_VERSIONS),
        "palsynet_protected_reads": 0,
        "mayo_reads": 0,
    }


def test_locked_registry_and_standard_threshold_are_closed(c):
    c.eq(LOCKED_DECISION_THRESHOLD, 0.5)
    c.eq(set(PROFILE_REGISTRY), {"neuroface", "meei"})
    c.eq(PROFILE_REGISTRY["neuroface"]["name"], "bilateral_range_fusion")
    c.eq(PROFILE_REGISTRY["meei"]["name"], "paired_action_expert_fusion")


def test_public_report_applies_all_profile_gate_without_row_evidence(c):
    report = build_public_report(_metrics(), _provenance())
    c.true(report["decision"]["all_profile_gate_passed"])
    c.eq(report["decision"]["primary_accuracy_floor"], 0.93)
    c.eq(report["decision"]["balanced_accuracy_floor"], 0.90)
    encoded = json.dumps(report, sort_keys=True)
    c.true("probability" not in encoded and "group_" not in encoded)
    c.true("rec_" not in encoded and "/Users/" not in encoded)


def test_gate_fails_if_one_accuracy_or_balanced_accuracy_misses(c):
    for profile, field, value, errors in (
        ("palsynet_development", "accuracy", 35 / 38, 3),
        ("neuroface", "balanced_accuracy", 0.899, 2),
    ):
        metrics = _metrics()
        metrics[profile][field] = value
        metrics[profile]["errors"] = errors
        report = build_public_report(metrics, _provenance())
        c.true(not report["decision"]["all_profile_gate_passed"])


def test_publication_is_no_overwrite_and_mode_0600(c):
    report = build_public_report(_metrics(), _provenance())
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "report.json"
        digest = write_public_report_no_overwrite(output, report)
        c.eq(len(digest), 64)
        c.eq(os.stat(output).st_mode & 0o777, 0o600)
        c.raises(lambda: write_public_report_no_overwrite(output, report), FileExistsError)


def test_evidence_reader_binds_exact_canonical_file_and_rejects_symlink(c):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        evidence = root / "evidence.bin"
        evidence.write_bytes(b"committed")
        c.eq(_read_exact_bytes(evidence, maximum=32), b"committed")
        link = root / "link.bin"
        link.symlink_to(evidence)
        c.raises(lambda: _read_exact_bytes(link, maximum=32), ValueError)


def test_json_evidence_rejects_duplicate_keys_and_nonobject(c):
    c.eq(_json_document(b'{"schema":"v1"}'), {"schema": "v1"})
    c.raises(lambda: _json_document(b'{"schema":"v1","schema":"v2"}'), ValueError)
    c.raises(lambda: _json_document(b"[]"), ValueError)


def test_report_rejects_runtime_dependency_drift(c):
    provenance = _provenance()
    provenance["runtime_versions"]["scikit_learn"] = "different"
    c.raises(lambda: build_public_report(_metrics(), provenance), ValueError)


if __name__ == "__main__":
    run_all("test_run_dense_action_router_v6", dict(globals()))
