"""Development-only comparison contracts for Action-Aligned 110D v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.action_aligned_110d_v1 import (  # noqa: E402
    CANDIDATE_ORDER,
    choose_locked_candidate,
    run_group_disjoint_oof,
)


def _fixture():
    rng = np.random.default_rng(41)
    labels = np.asarray([0, 0, 1, 1] * 4, dtype=np.int64)
    groups = np.asarray([f"group-{index}" for index in range(labels.size)])
    folds = np.repeat(np.arange(4), 4)
    baseline = rng.normal(size=(labels.size, 110))
    action = baseline.copy()
    signal = np.where(labels == 1, 2.0, -2.0)
    action[:, 0] = signal
    return labels, groups, folds, baseline, baseline.copy(), action, action.copy()


def test_registry_and_locking_rule_are_frozen(c: Check):
    c.eq(CANDIDATE_ORDER, ("four_time_window_110d", "seven_action_window_110d"))
    base = {"auroc": 0.9, "balanced_accuracy": 0.8, "brier": 0.2}
    improved = {"auroc": 0.9, "balanced_accuracy": 0.85, "brier": 0.19}
    c.eq(choose_locked_candidate({CANDIDATE_ORDER[0]: base, CANDIDATE_ORDER[1]: improved}), CANDIDATE_ORDER[1])
    worse = dict(improved, auroc=0.89)
    c.eq(choose_locked_candidate({CANDIDATE_ORDER[0]: base, CANDIDATE_ORDER[1]: worse}), CANDIDATE_ORDER[0])


def test_oof_is_group_disjoint_once_per_recording_and_fixed_model(c: Check):
    labels, groups, folds, baseline, baseline_mirror, action, action_mirror = _fixture()
    result = run_group_disjoint_oof(
        labels=labels,
        group_ids=groups,
        inner_folds=folds,
        original={CANDIDATE_ORDER[0]: baseline, CANDIDATE_ORDER[1]: action},
        mirrored={CANDIDATE_ORDER[0]: baseline_mirror, CANDIDATE_ORDER[1]: action_mirror},
    )
    c.eq(tuple(result.probabilities), CANDIDATE_ORDER)
    c.true(all(values.shape == labels.shape for values in result.probabilities.values()))
    c.true(all(np.isfinite(values).all() for values in result.probabilities.values()))
    c.eq(result.audit["development_model_fits"], 8)
    c.eq(result.audit["development_predictions"], 32)
    c.eq(result.audit["protected_feature_reads"], 0)
    c.eq(result.audit["protected_fits"], 0)
    c.eq(result.audit["protected_predictions"], 0)


def test_oof_rejects_group_or_matrix_drift(c: Check):
    labels, groups, folds, baseline, baseline_mirror, action, action_mirror = _fixture()
    groups[1] = groups[0]
    labels[1] = 1
    c.raises(lambda: run_group_disjoint_oof(
        labels=labels,
        group_ids=groups,
        inner_folds=folds,
        original={CANDIDATE_ORDER[0]: baseline, CANDIDATE_ORDER[1]: action},
        mirrored={CANDIDATE_ORDER[0]: baseline_mirror, CANDIDATE_ORDER[1]: action_mirror},
    ), ValueError)


if __name__ == "__main__":
    run_all("test_action_aligned_110d_v1", dict(globals()))
