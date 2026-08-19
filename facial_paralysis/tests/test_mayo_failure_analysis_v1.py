"""Contracts for Mayo failure analysis and PalsyNet-only robust inference."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.mayo_failure_analysis_v1 import (  # noqa: E402
    AGGREGATOR_ORDER,
    aggregate_mirror_probabilities,
    build_failure_summary,
    build_robust_inference_report,
    feature_region_assignments,
    select_palsynet_aggregator,
)
from scripts.run_mayo_failure_analysis_v1 import _write_no_overwrite  # noqa: E402
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_names,
)


def test_three_aggregators_are_frozen_and_deterministic(c: Check):
    c.eq(AGGREGATOR_ORDER, (
        "mirror_mean", "mirror_logit_mean", "mirror_conservative_max",
    ))
    first = np.asarray([0.2, 0.8, 0.4], dtype=np.float64)
    second = np.asarray([0.6, 0.2, 0.4], dtype=np.float64)
    c.true(np.allclose(
        aggregate_mirror_probabilities(first, second, "mirror_mean"),
        [0.4, 0.5, 0.4],
    ))
    c.true(np.allclose(
        aggregate_mirror_probabilities(
            first, second, "mirror_conservative_max"
        ),
        [0.6, 0.8, 0.4],
    ))
    swapped = aggregate_mirror_probabilities(
        second, first, "mirror_logit_mean"
    )
    original = aggregate_mirror_probabilities(
        first, second, "mirror_logit_mean"
    )
    c.true(np.array_equal(swapped, original), "aggregation is mirror-order invariant")
    c.raises(
        lambda: aggregate_mirror_probabilities(first, second[:-1], "mirror_mean"),
        ValueError,
    )
    c.raises(
        lambda: aggregate_mirror_probabilities(first, second, "unknown"),
        ValueError,
    )


def test_all_110_features_have_exact_clinical_regions(c: Check):
    names = candidate_feature_names(LANDMARK_MI_110D)
    regions = feature_region_assignments(names)
    c.eq(len(names), 110)
    c.eq(len(regions), 110)
    c.eq(set(regions), {"eye", "brow", "mouth"})
    c.raises(
        lambda: feature_region_assignments(names[:-1] + ("unknown_feature",)),
        ValueError,
        "unknown geometry cannot be silently assigned",
    )


def test_selection_uses_palsynet_metrics_and_retains_ties(c: Check):
    baseline = {
        "auroc": 0.95, "balanced_accuracy": 0.90, "brier": 0.12,
    }
    tied = {name: dict(baseline) for name in AGGREGATOR_ORDER}
    decision = select_palsynet_aggregator(tied)
    c.eq(decision["selected"], "mirror_mean")
    c.eq(decision["promoted"], False)

    improved = {name: dict(baseline) for name in AGGREGATOR_ORDER}
    improved["mirror_logit_mean"]["brier"] = 0.11
    improved["mirror_conservative_max"].update({
        "balanced_accuracy": 0.91, "brier": 0.14,
    })
    decision = select_palsynet_aggregator(improved)
    c.eq(decision["selected"], "mirror_logit_mean")
    c.eq(decision["promoted"], True)
    c.eq(
        decision["eligibility"]["mirror_conservative_max"], False,
        "Brier degradation beyond 0.01 blocks promotion",
    )


def test_failure_summary_is_aggregate_identifier_free_and_one_class(c: Check):
    scores = np.asarray([0.4, 0.48, 0.7, 0.9], dtype=np.float64)
    coverage = np.asarray([0.9, 1.0, 1.0, 1.0], dtype=np.float64)
    nuisance_names = ("luminance_mean", "face_scale_mean")
    nuisance = np.asarray([
        [80.0, 0.20], [90.0, 0.22], [100.0, 0.25], [120.0, 0.30],
    ])
    contributions = np.asarray([
        [-0.4, -0.2, 0.1], [-0.3, -0.1, 0.2],
        [0.2, 0.1, 0.4], [0.4, 0.2, 0.5],
    ])
    report = build_failure_summary(
        scores, coverage, nuisance, nuisance_names, contributions,
    )
    encoded = json.dumps(report, sort_keys=True).lower()
    c.eq(report["counts"]["below_threshold"], 2)
    c.eq(report["claim_scope"], "mayo_assumed_positive_aggregate_diagnostic")
    c.eq(report["accuracy_defined"], False)
    for forbidden in ("recording_id", "group_id", "source_sha256", ".mov", "/users/"):
        c.true(forbidden not in encoded)
    c.eq(set(report["region_logit_contribution_shift"]), {"eye", "brow", "mouth"})


def test_final_report_rejects_protected_access_and_private_identifiers(c: Check):
    metrics = {
        name: {"auroc": 0.95, "balanced_accuracy": 0.9, "brier": 0.1}
        for name in AGGREGATOR_ORDER
    }
    decision = select_palsynet_aggregator(metrics)
    failure = {
        "schema_version": "mayo_failure_analysis_v1_aggregate",
        "claim_scope": "mayo_assumed_positive_aggregate_diagnostic",
        "accuracy_defined": False,
        "mayo_used_for_model_selection": False,
    }
    positive = {
        "records": 47, "positive_calls": 45, "positive_call_rate": 45 / 47,
        "accuracy_defined": False,
    }
    provenance = {name: char * 64 for name, char in zip((
        "palsynet_source_collection_sha256",
        "palsynet_reviewed_manifest_sha256",
        "palsynet_review_ledger_sha256",
        "palsynet_split_registry_sha256",
        "mayo_cache_manifest_sha256",
        "implementation_sha256",
    ), "abcdef")}
    report = build_robust_inference_report(
        metrics, decision, failure, positive, positive,
        development_recordings=39,
        development_groups=38,
        mayo_records=47,
        provenance=provenance,
        protected_cache_records_loaded=0,
    )
    encoded = json.dumps(report, sort_keys=True).lower()
    c.eq(report["decision"]["mayo_used_for_model_selection"], False)
    c.eq(report["audit"]["protected_predictions"], 0)
    for forbidden in ("recording_id", "group_id", "source_sha256", ".mov", "/users/"):
        c.true(forbidden not in encoded)
    c.raises(
        lambda: build_robust_inference_report(
            metrics, decision, failure, positive, positive,
            development_recordings=39,
            development_groups=38,
            mayo_records=47,
            provenance=provenance,
            protected_cache_records_loaded=1,
        ),
        ValueError,
        "any protected cache access invalidates the report",
    )


def test_report_writer_is_private_and_refuses_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "report.json"
        _write_no_overwrite(path, {"ok": True})
        c.eq(os.stat(path).st_mode & 0o777, 0o600)
        c.raises(
            lambda: _write_no_overwrite(path, {"ok": False}),
            FileExistsError,
        )


if __name__ == "__main__":
    run_all("test_mayo_failure_analysis_v1", dict(globals()))
