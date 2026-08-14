"""Closed contracts for frozen 110D NeuroFace transfer evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_neuroface_external_v1 import _parser  # noqa: E402
from src.evaluation.neuroface_external_v1 import (  # noqa: E402
    ExternalAudit,
    aggregate_participant_scores,
    build_expected_authorization,
    build_external_report,
    canonical_json_sha256,
    metric_report,
    validate_external_authorization,
    validate_external_report,
)
from _testlib import Check, run_all  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _authorization_inputs() -> dict[str, object]:
    return {
        "preanalysis_registration_sha256": _sha("registration"),
        "final_artifact_sha256": _sha("artifact"),
        "private_manifest_sha256": _sha("private"),
        "cache_manifest_sha256": _sha("cache-manifest"),
        "cache_artifact_collection_sha256": _sha("cache-artifacts"),
        "implementation_sha256": _sha("implementation"),
        "dependency_lock_sha256": _sha("environment"),
        "expected_participants": 36,
        "expected_affected": 25,
        "expected_unaffected": 11,
        "expected_videos": 261,
    }


def test_authorization_is_exact_and_cli_cannot_tune(c: Check):
    inputs = _authorization_inputs()
    authorization = build_expected_authorization(**inputs)
    digest = canonical_json_sha256(authorization)
    audit = ExternalAudit()
    state = validate_external_authorization(
        authorization,
        authorization_sha256=digest,
        pinned_authorization_sha256=digest,
        audit=audit,
        **inputs,
    )
    c.eq(state.authorization_sha256, digest, "out-of-band pin authenticates exact run")
    c.eq(audit.authorization_passes, 1, "authorization passes once")
    changed = copy.deepcopy(authorization)
    changed["protocol"]["threshold"] = 0.4
    c.raises(lambda: validate_external_authorization(
        changed,
        authorization_sha256=canonical_json_sha256(changed),
        pinned_authorization_sha256=digest,
        audit=ExternalAudit(),
        **inputs,
    ), ValueError, "threshold forgery is rejected")
    options = {action.dest for action in _parser()._actions}
    c.eq(options, {
        "help", "final_artifact", "private_manifest", "feature_cache_root",
        "preanalysis_registration", "dependency_lock", "authorization",
    }, "runner exposes inputs only and no tuning or output controls")


def _video_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant, cohort, label, base in (
        ("grp_" + "1" * 64, "healthy_control", 0, 0.1),
        ("grp_" + "2" * 64, "als", 1, 0.7),
        ("grp_" + "3" * 64, "post_stroke", 1, 0.8),
        ("grp_" + "4" * 64, "healthy_control", 0, 0.2),
    ):
        for offset, task in enumerate(("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")):
            rows.append({
                "participant_id": participant,
                "cohort": cohort,
                "label": label,
                "task": task,
                "probability": base + offset * 0.03,
            })
        rows.append({
            "participant_id": participant,
            "cohort": cohort,
            "label": label,
            "task": "BBP_NORMAL",
            "probability": min(1.0, base + 0.09),
        })
    return rows


def test_primary_aggregation_uses_exact_three_common_tasks(c: Check):
    aggregate = aggregate_participant_scores(_video_rows())
    c.eq(aggregate.labels.tolist(), [0, 1, 1, 0], "participant ordering is stable")
    c.true(bool(np.allclose(aggregate.primary_scores, [0.13, 0.73, 0.83, 0.23])),
           "primary score is the exact unweighted mean of three common tasks")
    c.true(bool(np.allclose(aggregate.all_task_scores, [0.145, 0.745, 0.845, 0.245])),
           "all-task mean is retained only as a secondary endpoint")
    missing = _video_rows()[:-2]
    c.raises(lambda: aggregate_participant_scores(missing), ValueError,
             "missing a primary task invalidates the closed primary cohort")


def test_metrics_and_report_are_participant_level_and_identifier_free(c: Check):
    aggregate = aggregate_participant_scores(_video_rows())
    metrics = metric_report(
        aggregate.labels, aggregate.primary_scores, aggregate.cohorts, repeats=32
    )
    c.eq(metrics["accuracy"]["point"], 1.0, "ordinary accuracy is explicitly reported")
    c.eq(metrics["balanced_accuracy"]["point"], 1.0,
         "balanced accuracy is independently reported")
    inputs = _authorization_inputs()
    authorization = build_expected_authorization(**inputs)
    digest = canonical_json_sha256(authorization)
    audit = ExternalAudit(authorization_attempts=1, authorization_passes=1,
                          cache_artifacts_hashed=16, cache_records_loaded=16,
                          feature_extractions=16, mirror_transforms=32,
                          artifact_predictions=16, participant_aggregations=4)
    state = validate_external_authorization(
        authorization,
        authorization_sha256=digest,
        pinned_authorization_sha256=digest,
        audit=ExternalAudit(),
        **inputs,
    )
    # Preserve the synthetic execution counters while using the authenticated state.
    report = build_external_report(
        aggregate,
        state=state,
        audit=audit,
        bootstrap_repeats=32,
        provenance={"final_artifact_sha256": _sha("artifact")},
    )
    validate_external_report(
        report, aggregate=aggregate, state=state, audit=audit,
        expected_bootstrap_repeats=32,
    )
    encoded = json.dumps(report, allow_nan=False)
    c.true(all(token not in encoded for token in (
        "grp_", "rec_", "participant_id", "probability", "/Users/", ".avi"
    )), "public report contains aggregate statistics only")
    tampered = copy.deepcopy(report)
    tampered["endpoints"]["primary_three_task"]["metrics"]["auroc"]["point"] = 0.0
    c.raises(lambda: validate_external_report(
        tampered, aggregate=aggregate, state=state, audit=audit,
        expected_bootstrap_repeats=32,
    ), ValueError, "metric changes fail independent recomputation")


if __name__ == "__main__":
    run_all("test_neuroface_external_v1", dict(globals()))
