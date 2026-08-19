from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402
from scripts.run_selective_universal_router_v5 import (  # noqa: E402
    build_aggregate_report,
    load_private_profile_bytes,
    publish_report,
)
from src.evaluation.selective_router_v5 import (  # noqa: E402
    EVIDENCE_PROFILES,
    evaluate_profile,
)


def _payload(profile: str, *, invert: bool = False) -> bytes:
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    probability = np.asarray([
        0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45,
        0.55, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99,
    ], dtype=np.float64)
    if invert:
        probability = 1.0 - probability
    buffer = io.BytesIO()
    np.savez(
        buffer,
        schema_version=np.asarray("selective_router_v5_private_profile"),
        evidence_profile=np.asarray(profile),
        anonymous_groups=np.asarray([f"anon_{i:02d}" for i in range(20)]),
        labels=labels,
        final_probability=probability,
        component_probability=np.stack((probability, probability), axis=1),
        decision_threshold=np.full(20, 0.5, dtype=np.float64),
    )
    return buffer.getvalue()


def _inputs(*, invert_profile: str | None = None):
    payloads = {
        profile: _payload(profile, invert=profile == invert_profile)
        for profile in EVIDENCE_PROFILES
    }
    model = b'{"schema_version":"universal_clinical_router_v4"}\n'
    evaluations = {}
    mapping = {
        "free_asymmetry": "palsynet_development",
        "scripted_multimechanism": "neuroface_development",
        "cue_aligned_upper": "meei_development",
    }
    for profile, name in mapping.items():
        private = load_private_profile_bytes(payloads[profile])
        baseline = evaluate_profile(private)["baseline"]
        evaluations[name] = {
            "participants": 20,
            "metrics": {
                key: baseline[key]
                for key in (
                    "accuracy", "balanced_accuracy", "sensitivity", "specificity"
                )
            },
        }
    v4_report = {
        "schema_version": "universal_clinical_router_v4_aggregate_report",
        "model_artifact": {"sha256": hashlib.sha256(model).hexdigest()},
        "evaluations": evaluations,
    }
    v4_report_bytes = (json.dumps(v4_report, sort_keys=True) + "\n").encode()
    boundary = {
        "schema_version": "selective_router_v5_boundary_attestation",
        "protected_palsynet_reads": 0,
        "palsynet_sealed_outer_reads": 0,
        "mayo_reads": 0,
        "profile_generator_sha256": "a" * 64,
        "helper_aggregate_sha256": "b" * 64,
        "profile_payload_sha256": {
            profile: hashlib.sha256(payload).hexdigest()
            for profile, payload in payloads.items()
        },
    }
    return payloads, model, v4_report_bytes, boundary


def test_private_npz_loader_is_closed_and_exact(c: Check):
    payload = _payload("free_asymmetry")
    document = load_private_profile_bytes(payload)
    c.eq(document["evidence_profile"], "free_asymmetry")
    c.true(type(document["anonymous_groups"]) is tuple)
    malformed = io.BytesIO()
    np.savez(malformed, surprise=np.asarray(1))
    c.raises(lambda: load_private_profile_bytes(malformed.getvalue()), ValueError,
             "unknown NPZ members fail closed")
    c.raises(lambda: load_private_profile_bytes(b"PK\x03\x04"), ValueError,
             "truncated ZIP containers fail closed")


def test_aggregate_report_binds_v4_and_contains_no_row_evidence(c: Check):
    payloads, model, v4_report, boundary = _inputs()
    report = build_aggregate_report(
        payloads, v4_model_bytes=model, v4_report_bytes=v4_report,
        boundary_attestation=boundary,
    )
    c.eq(report["v4_model_sha256"], hashlib.sha256(model).hexdigest())
    c.eq(report["v4_report_sha256"], hashlib.sha256(v4_report).hexdigest())
    c.eq(report["audit"]["profile_generator_sha256"], "a" * 64)
    c.eq(report["audit"]["helper_aggregate_sha256"], "b" * 64)
    c.true(report["decision"]["passed"])
    encoded = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "anonymous_groups", "labels", "final_probability",
        "component_probability", "selection_sha256", "/users/", "/home/",
    ):
        c.true(forbidden not in encoded, f"public output excludes {forbidden}")


def test_boundary_and_profile_hashes_fail_closed(c: Check):
    payloads, model, v4_report, boundary = _inputs()
    bad = dict(boundary, mayo_reads=1)
    c.raises(
        lambda: build_aggregate_report(
            payloads, v4_model_bytes=model, v4_report_bytes=v4_report,
            boundary_attestation=bad,
        ), ValueError, "Mayo reads are forbidden",
    )
    drifted = dict(boundary)
    drifted["profile_payload_sha256"] = dict(boundary["profile_payload_sha256"])
    drifted["profile_payload_sha256"]["free_asymmetry"] = "0" * 64
    c.raises(
        lambda: build_aggregate_report(
            payloads, v4_model_bytes=model, v4_report_bytes=v4_report,
            boundary_attestation=drifted,
        ), ValueError, "private profile bytes must match the boundary evidence",
    )


def test_v4_baseline_mismatch_is_rejected(c: Check):
    payloads, model, v4_report, boundary = _inputs()
    report = json.loads(v4_report)
    report["evaluations"]["meei_development"]["metrics"]["accuracy"] = 0.123
    forged = (json.dumps(report, sort_keys=True) + "\n").encode()
    c.raises(
        lambda: build_aggregate_report(
            payloads, v4_model_bytes=model, v4_report_bytes=forged,
            boundary_attestation=boundary,
        ), ValueError, "selective evidence must reproduce v4 before evaluation",
    )


def test_report_publication_is_no_overwrite(c: Check):
    payloads, model, v4_report, boundary = _inputs()
    report = build_aggregate_report(
        payloads, v4_model_bytes=model, v4_report_bytes=v4_report,
        boundary_attestation=boundary,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.json"
        digest = publish_report(path, report)
        c.eq(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        original = path.read_bytes()
        c.raises(lambda: publish_report(path, report), FileExistsError,
                 "formal aggregate output cannot be overwritten")
        c.eq(path.read_bytes(), original)


if __name__ == "__main__":
    run_all("test_run_selective_universal_router_v5", dict(globals()))
