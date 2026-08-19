from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.selective_router_v5 import (  # noqa: E402
    CANDIDATE_ORDER,
    COVERAGES,
    PRIMARY_COVERAGE,
    confidence_scores,
    evaluate_profile,
    select_candidate,
)


def _profile() -> dict[str, object]:
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    final = np.asarray([
        0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45,
        0.55, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99,
    ], dtype=np.float64)
    components = np.stack((
        final,
        np.asarray([
            0.02, 0.04, 0.06, 0.10, 0.12, 0.18, 0.22, 0.35, 0.55, 0.60,
            0.40, 0.45, 0.72, 0.82, 0.87, 0.91, 0.93, 0.96, 0.98, 0.995,
        ], dtype=np.float64),
    ), axis=1)
    return {
        "schema_version": "selective_router_v5_private_profile",
        "evidence_profile": "free_asymmetry",
        "anonymous_groups": tuple(f"g{index:02d}" for index in range(20)),
        "labels": labels,
        "final_probability": final,
        "component_probability": components,
        "decision_threshold": np.full(20, 0.5, dtype=np.float64),
    }


def test_four_confidence_scores_are_exact_and_immutable(c: Check):
    final = np.asarray([0.1, 0.8], dtype=np.float64)
    components = np.asarray([[0.1, 0.2], [0.6, 0.9]], dtype=np.float64)
    result = confidence_scores(
        final, components,
        decision_threshold=np.asarray([0.5, 0.5], dtype=np.float64),
    )
    c.eq(tuple(result), CANDIDATE_ORDER, "candidate order is frozen")
    c.true(np.allclose(result["probability_margin"], [0.4, 0.3]))
    c.true(np.allclose(result["range_penalized_margin"], [0.35, 0.15]))
    c.true(np.allclose(result["unanimous_min_margin"], [0.3, 0.1]))
    expected = np.asarray([0.4 / 1.2, 0.3 / 1.6])
    c.true(np.allclose(result["dispersion_normalized_margin"], expected))
    c.raises(
        lambda: result["probability_margin"].setflags(write=True), ValueError,
        "confidence vectors are immutable snapshots",
    )


def test_disagreement_invalidates_unanimous_confidence(c: Check):
    result = confidence_scores(
        np.asarray([0.8], dtype=np.float64),
        np.asarray([[0.9, 0.4]], dtype=np.float64),
        decision_threshold=np.asarray([0.5], dtype=np.float64),
    )
    c.true(result["unanimous_min_margin"][0] < 0.0,
           "one disagreeing component forces negative confidence")


def test_profile_schema_is_closed_and_source_blind(c: Check):
    malformed = dict(_profile(), source="NeuroFace")
    c.raises(lambda: evaluate_profile(malformed), ValueError,
             "source or dataset fields cannot enter the evaluator")
    malformed = dict(_profile())
    malformed["final_probability"] = malformed["final_probability"].astype(np.float32)
    c.raises(lambda: evaluate_profile(malformed), ValueError,
             "numeric schema is exact float64")


def test_top_coverage_is_deterministic_and_label_independent(c: Check):
    profile = _profile()
    first = evaluate_profile(profile)
    permuted = dict(profile, labels=1 - profile["labels"])
    second = evaluate_profile(permuted)
    for candidate in CANDIDATE_ORDER:
        for coverage in COVERAGES:
            a = first["candidates"][candidate][f"{coverage:.2f}"]
            b = second["candidates"][candidate][f"{coverage:.2f}"]
            c.eq(a["selection_sha256"], b["selection_sha256"],
                 "labels cannot change retained indices")
            c.eq(a["retained"], int(np.ceil(coverage * 20)))
    c.eq(PRIMARY_COVERAGE, 0.70)


def test_component_permutation_does_not_change_scores(c: Check):
    profile = _profile()
    reversed_profile = dict(
        profile,
        component_probability=profile["component_probability"][:, ::-1].copy(),
    )
    a = evaluate_profile(profile)
    b = evaluate_profile(reversed_profile)
    c.eq(a["candidates"], b["candidates"],
         "all registered consensus scores are symmetric across heads")


def test_selective_metrics_keep_coverage_errors_and_both_classes(c: Check):
    report = evaluate_profile(_profile())
    primary = report["candidates"]["probability_margin"]["0.70"]
    c.eq(primary["retained"], 14)
    c.eq(primary["abstained"], 6)
    c.eq(primary["errors"], 0)
    c.eq(primary["accuracy"], 1.0)
    c.eq(primary["balanced_accuracy"], 1.0)
    c.true(primary["retained_negative"] >= 5)
    c.true(primary["retained_positive"] >= 5)


def test_nested_fold_threshold_vector_controls_oof_decisions(c: Check):
    profile = _profile()
    final = profile["final_probability"].copy()
    final[0], final[-1] = 0.51, 0.49
    threshold = profile["decision_threshold"].copy()
    threshold[0], threshold[-1] = 0.60, 0.40
    profile["final_probability"] = final
    profile["component_probability"] = np.stack((final, final), axis=1)
    profile["decision_threshold"] = threshold
    report = evaluate_profile(profile)
    c.eq(report["baseline"]["errors"], 0,
         "per-participant nested thresholds define OOF predictions")
    c.eq(report["decision_threshold_scope"], "per_participant_oof")


def test_candidate_gate_requires_every_profile(c: Check):
    passed = {
        profile: evaluate_profile(dict(_profile(), evidence_profile=profile))
        for profile in (
            "free_asymmetry", "scripted_multimechanism", "cue_aligned_upper"
        )
    }
    decision = select_candidate(passed)
    c.true(decision["passed"], "a candidate must pass every profile")
    c.eq(decision["selected"], "probability_margin",
         "registry order is the final deterministic tie break")
    failed = dict(passed)
    hard = _profile()
    hard["final_probability"] = 1.0 - hard["final_probability"]
    hard["component_probability"] = np.stack(
        (hard["final_probability"], hard["final_probability"]), axis=1
    )
    hard["evidence_profile"] = "cue_aligned_upper"
    failed["cue_aligned_upper"] = evaluate_profile(hard)
    rejected = select_candidate(failed)
    c.true(not rejected["passed"] and rejected["selected"] is None,
           "one failed evidence profile rejects the universal candidate")


if __name__ == "__main__":
    run_all("test_selective_router_v5", dict(globals()))
