"""Closed tests for participant-disjoint NeuroFace action-capacity experts."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from inspect import signature
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
import src.evaluation.neuroface_action_capacity_v1 as capacity_eval  # noqa: E402
from src.evaluation.neuroface_action_capacity_v1 import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    COHORTS,
    FROZEN_110D_DESCRIPTIVE_COMPARATOR,
    PRIMARY_TASKS,
    ActionCapacityDataset,
    ActionCapacityAudit,
    build_public_report,
    evaluate_action_capacity_oof,
    protocol,
    validate_public_report,
)


def _dataset(*, missing: bool = False, shuffle: bool = True) -> ActionCapacityDataset:
    rng = np.random.default_rng(814)
    cohorts = ("als",) * 11 + ("healthy_control",) * 11 + ("post_stroke",) * 14
    participants = tuple(f"grp_{index:064x}" for index in range(36))
    original = []
    mirrored = []
    groups = []
    row_cohorts = []
    tasks = []
    for group_index, (group, cohort) in enumerate(zip(participants, cohorts)):
        affected = float(cohort != "healthy_control")
        for task_index, task in enumerate(PRIMARY_TASKS):
            signal = np.zeros(18, dtype=np.float64)
            signal[task_index * 3:(task_index + 1) * 3] = (
                affected * 1.5 + rng.normal(0.0, 0.15, size=3)
            )
            original.append(signal)
            # Deliberately distinct, so probability averaging is observable.
            mirrored.append(signal + rng.normal(0.0, 0.03, size=18))
            groups.append(group)
            row_cohorts.append(cohort)
            tasks.append(task)
    if missing:
        original.pop()
        mirrored.pop()
        groups.pop()
        row_cohorts.pop()
        tasks.pop()
    order = np.arange(len(groups))
    if shuffle:
        rng.shuffle(order)
    return ActionCapacityDataset(
        original_features=np.asarray(original, dtype=np.float64)[order],
        mirrored_features=np.asarray(mirrored, dtype=np.float64)[order],
        participant_ids=np.asarray(groups, dtype=object)[order],
        tasks=np.asarray(tasks, dtype=object)[order],
        cohorts=np.asarray(row_cohorts, dtype=object)[order],
    )


def _provenance():
    return {
        "private_manifest_sha256": "a" * 64,
        "collection_manifest_sha256": "b" * 64,
        "primary_cache_collection_sha256": "c" * 64,
        "implementation_sha256": "d" * 64,
        "dependency_lock_sha256": "e" * 64,
        "mount_attestation_sha256": "f" * 64,
    }


def _runtime():
    return {"host_class": "nebius_h200", "device_class": "cuda", "seconds": 1.0}


def _validate_report(report, result):
    return validate_public_report(
        report,
        result=result,
        expected_provenance=_provenance(),
        expected_runtime=_runtime(),
        expected_audit=ActionCapacityAudit(),
    )


def test_protocol_is_exact_and_not_tunable(c: Check):
    c.eq(PRIMARY_TASKS, ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"))
    c.eq(COHORTS, ("als", "healthy_control", "post_stroke"))
    c.eq((BOOTSTRAP_REPEATS, BOOTSTRAP_SEED), (5000, 20260814))
    frozen = protocol()
    c.eq(frozen["folds"], {"count": 6, "seed": 20260813})
    c.eq(frozen["expert"], {
        "type": "standardized_l2_logistic",
        "C": 0.01,
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": 0,
        "threshold": 0.5,
    })
    c.eq(frozen["training_mirror_weights"], {
        "original": 0.5, "horizontal_mirror": 0.5,
        "total_per_participant_per_task": 1.0,
    })
    c.eq(frozen["participant_score"],
         "mean_of_three_task_scores_after_each_original_mirror_mean")
    c.eq(tuple(signature(evaluate_action_capacity_oof).parameters), ("dataset",))


def test_oof_is_group_disjoint_and_uses_one_fixed_expert_per_task(c: Check):
    dataset = _dataset()
    result = capacity_eval._evaluate_action_capacity_oof_fixture(
        dataset, bootstrap_repeats=32
    )
    c.eq(result.participant_scores.shape, (36,))
    c.eq(result.task_scores.shape, (36, 3))
    c.eq(result.original_probabilities.shape, (36, 3))
    c.eq(result.mirrored_probabilities.shape, (36, 3))
    c.true(np.allclose(
        result.task_scores,
        (result.original_probabilities + result.mirrored_probabilities) / 2.0,
        atol=0.0, rtol=0.0,
    ), "original and mirror probabilities are averaged before tasks")
    c.true(np.allclose(
        result.participant_scores, result.task_scores.mean(axis=1),
        atol=0.0, rtol=0.0,
    ), "three task probabilities are unweighted")
    c.eq(set(result.fold_assignments.tolist()), set(range(6)))
    c.eq(len(result.fit_audit), 18)
    for fit in result.fit_audit:
        c.true(set(fit.training_cohorts) == set(COHORTS))
        c.eq(fit.training_rows, fit.training_participants * 2)
        c.eq(fit.original_weight_sum, fit.training_participants * 0.5)
        c.eq(fit.mirrored_weight_sum, fit.training_participants * 0.5)
        c.eq(fit.total_weight, float(fit.training_participants))
        c.eq((fit.C, fit.solver, fit.max_iter, fit.random_state),
             (0.01, "liblinear", 2000, 0))
        held = set(result.participant_ids[result.fold_assignments == fit.fold].tolist())
        expected_train = frozenset(
            result.participant_ids[result.fold_assignments != fit.fold].tolist()
        )
        c.eq(fit.training_participant_ids, expected_train)
        c.true(not held.intersection(fit.training_participant_ids),
               "held-out participants never enter the task expert")


def test_order_is_deterministic_and_labels_come_only_from_cohort(c: Check):
    first = capacity_eval._evaluate_action_capacity_oof_fixture(
        _dataset(shuffle=True), bootstrap_repeats=24
    )
    second = capacity_eval._evaluate_action_capacity_oof_fixture(
        _dataset(shuffle=False), bootstrap_repeats=24
    )
    c.true(np.array_equal(first.participant_ids, second.participant_ids))
    c.true(np.array_equal(first.fold_assignments, second.fold_assignments))
    c.true(np.allclose(first.participant_scores, second.participant_scores,
                       atol=1e-15, rtol=0.0))
    expected = np.asarray(
        [int(value != "healthy_control") for value in first.cohorts],
        dtype=np.int64,
    )
    c.true(np.array_equal(first.labels, expected))
    c.eq(int(np.sum(first.cohorts == "als")), 11)
    c.eq(int(np.sum(first.cohorts == "healthy_control")), 11)
    c.eq(int(np.sum(first.cohorts == "post_stroke")), 14)


def test_missing_task_bad_cohort_or_nonfinite_feature_fails_closed(c: Check):
    c.raises(lambda: capacity_eval._evaluate_action_capacity_oof_fixture(
        _dataset(missing=True), bootstrap_repeats=8
    ), ValueError, "one missing primary task causes abstention/failure")

    bad_cohort = _dataset()
    cohorts = bad_cohort.cohorts.copy()
    cohorts[0] = "bell_palsy"
    c.raises(lambda: capacity_eval._evaluate_action_capacity_oof_fixture(
        ActionCapacityDataset(
            bad_cohort.original_features, bad_cohort.mirrored_features,
            bad_cohort.participant_ids, bad_cohort.tasks, cohorts,
        ), bootstrap_repeats=8
    ), ValueError)

    bad_feature = _dataset()
    values = bad_feature.original_features.copy()
    values[0, 0] = np.nan
    c.raises(lambda: capacity_eval._evaluate_action_capacity_oof_fixture(
        ActionCapacityDataset(
            values, bad_feature.mirrored_features, bad_feature.participant_ids,
            bad_feature.tasks, bad_feature.cohorts,
        ), bootstrap_repeats=8
    ), ValueError)


def test_bootstrap_and_public_report_are_cohort_stratified_and_identifier_free(c: Check):
    fixture = capacity_eval._evaluate_action_capacity_oof_fixture(
        _dataset(), bootstrap_repeats=40
    )
    c.eq(fixture.bootstrap["repeats"], 40)
    c.eq(fixture.bootstrap["seed"], 20260814)
    c.eq(fixture.bootstrap["draw_sizes"], {
        "als": 11, "healthy_control": 11, "post_stroke": 14,
    })
    c.true(fixture.bootstrap["valid_draws"] >= 38)
    for name in (
        "auroc", "average_precision", "brier", "balanced_accuracy",
        "sensitivity", "specificity",
    ):
        c.true(0.0 <= fixture.metrics[name] <= 1.0)
        interval = fixture.bootstrap["intervals"][name]
        c.true(0.0 <= interval["lower"] <= interval["upper"] <= 1.0)

    result = evaluate_action_capacity_oof(_dataset())
    c.eq(result.bootstrap["repeats"], 5000)
    c.eq(result.bootstrap["minimum_valid_draws"], 4750)
    report = build_public_report(
        result,
        provenance=_provenance(),
        audit=ActionCapacityAudit(),
        runtime=_runtime(),
    )
    _validate_report(report, result)
    c.eq(report["descriptive_comparator"], FROZEN_110D_DESCRIPTIVE_COMPARATOR)
    encoded = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "grp_", "rec_", "participant_id", "recording_id", "/users/",
        "/home/", "palsynet_root", "cache_root",
    ):
        c.true(forbidden not in encoded, f"public report excludes {forbidden}")
    c.eq(report["audit"]["palsynet_path_accesses"], 0)
    c.eq(report["audit"]["palsynet_cache_reads"], 0)
    c.eq(report["audit"]["palsynet_predictions"], 0)
    c.eq(report["decision"]["current_110d_replaced"], False)
    c.eq(report["decision"]["fusion_authorized"], False)
    c.eq(report["decision"]["mayo_accuracy_claim_authorized"], False)

    tampered = copy.deepcopy(report)
    tampered["audit"]["palsynet_path_accesses"] = 1
    c.raises(lambda: _validate_report(tampered, result), ValueError)

    tampered_provenance = copy.deepcopy(report)
    tampered_provenance["provenance"]["implementation_sha256"] = "f" * 64
    c.raises(lambda: _validate_report(
        tampered_provenance, result
    ), ValueError, "self-reported legal SHA cannot replace out-of-band provenance")

    one_draw_fixture = capacity_eval._evaluate_action_capacity_oof_fixture(
        _dataset(), bootstrap_repeats=1
    )
    forged_short = replace(result, bootstrap=one_draw_fixture.bootstrap)
    c.raises(lambda: build_public_report(
        forged_short, provenance=_provenance(), audit=ActionCapacityAudit(),
        runtime=_runtime(),
    ), ValueError, "a short fixture bootstrap cannot become a formal v1 report")
    c.raises(lambda: build_public_report(
        one_draw_fixture, provenance=_provenance(), audit=ActionCapacityAudit(),
        runtime=_runtime(),
    ), ValueError, "fixture result type cannot become a formal v1 report")
    one_draw_report = copy.deepcopy(report)
    one_draw_report["protocol"]["bootstrap"]["repeats"] = 1
    one_draw_report["protocol"]["bootstrap"]["minimum_valid_draws"] = 1
    one_draw_report["bootstrap"]["repeats"] = 1
    one_draw_report["bootstrap"]["minimum_valid_draws"] = 1
    one_draw_report["bootstrap"]["valid_draws"] = 1
    c.raises(lambda: _validate_report(
        one_draw_report, result
    ), ValueError, "a one-draw report cannot validate as formal v1")

    forged_task = replace(
        result,
        per_task_metrics={
            **result.per_task_metrics,
            "NSM_KISS": {
                "auroc": 0.123456,
                "coverage_participants": 36,
                "coverage_fraction": 1.0,
            },
        },
    )
    c.raises(lambda: build_public_report(
        forged_task, provenance=_provenance(), audit=ActionCapacityAudit(),
        runtime=_runtime(),
    ), ValueError, "formal report recomputes per-task AUROC")

    tampered_ci = copy.deepcopy(report)
    tampered_ci["metrics"]["auroc"]["ci95"] = {"lower": 0.99, "upper": 0.99}
    c.raises(lambda: _validate_report(
        tampered_ci, result
    ), ValueError, "formal validator recomputes the fixed-seed bootstrap")

    for leaked in ("grp_" + "1" * 64, "/home/private/cache.npz", "participant_id"):
        tampered = copy.deepcopy(report)
        tampered["decision"]["criterion"] = leaked
        try:
            _validate_report(tampered, result)
        except ValueError as exc:
            c.true("identifier, path, or secret" in str(exc),
                   "recursive sensitive-content scan runs before schema checks")
        else:
            raise AssertionError("sensitive public report content was accepted")


if __name__ == "__main__":
    run_all("test_neuroface_action_capacity_v1", dict(globals()))
