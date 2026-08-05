"""Fixed proportional-odds model and subject-disjoint YFP evaluation."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.l2_cumulative_logit import (  # noqa: E402
    BOOTSTRAP_ATTEMPT_LIMIT,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    FIXED_C,
    OPTIMIZER_OPTIONS,
    PROBABILITY_FLOOR,
    L2CumulativeLogit,
    OrdinalModelError,
    cumulative_probabilities,
    group_oof_probabilities,
    group_total_one_weights,
    initial_parameters,
    ordered_cutpoints,
    proportional_odds_objective,
    subject_cluster_bootstrap,
    weighted_ordinal_metrics,
)
from _testlib import Check, run_all  # noqa: E402


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([f"g{i}" for i in range(9) for _ in range(2)])
    y = np.asarray([grade for grade in (0, 1, 2) for _group in range(3) for _ in range(2)])
    x = np.zeros((18, 6), dtype=np.float64)
    x[:, 0] = y + np.linspace(-0.1, 0.1, 18)
    x[:, 1] = np.arange(18) % 2
    return x, y, groups


def test_fixed_protocol_has_no_tuning_surface(c: Check):
    c.eq(FIXED_C, 0.01)
    c.eq(PROBABILITY_FLOOR, 1e-12)
    c.eq(OPTIMIZER_OPTIONS, {"maxiter": 2000, "ftol": 1e-12,
                              "gtol": 1e-8, "maxls": 50})
    c.eq((BOOTSTRAP_REPEATS, BOOTSTRAP_ATTEMPT_LIMIT, BOOTSTRAP_SEED),
         (5000, 100000, 20260805))
    c.raises(lambda: L2CumulativeLogit(C=1.0), TypeError, "C cannot be tuned")


def test_probability_equation_ordered_gap_and_initialization(c: Check):
    beta = np.asarray([0.5, -0.25])
    x = np.asarray([[2.0, 4.0], [0.0, 0.0]])
    raw_gap = float(np.log(np.expm1(1.0 - 1e-6)))
    theta = ordered_cutpoints(-0.5, raw_gap)
    c.true(abs((theta[1] - theta[0]) - 1.0) < 1e-12)
    expected = 1.0 / (1.0 + np.exp(-(theta[None, :] - x @ beta[:, None])))
    c.true(np.allclose(cumulative_probabilities(x, beta, theta), expected))
    init = initial_parameters(2)
    c.true(np.allclose(init[:2], 0.0))
    c.true(abs(init[2] + 0.5) < 1e-15)
    c.true(abs(ordered_cutpoints(init[2], init[3])[1] - 0.5) < 1e-12)


def test_objective_is_group_weighted_sum_nll_plus_beta_penalty(c: Check):
    x = np.asarray([[0.0], [0.0], [1.0]])
    y = np.asarray([0, 1, 2])
    groups = np.asarray(["a", "a", "b"])
    weights = group_total_one_weights(groups)
    c.true(np.allclose(weights, [0.5, 0.5, 1.0]))
    params = initial_parameters(1)
    value = proportional_odds_objective(params, x, y, weights)
    beta = params[0]
    theta = ordered_cutpoints(params[1], params[2])
    cum = cumulative_probabilities(x, np.asarray([beta]), theta)
    probs = np.column_stack((cum[:, 0], cum[:, 1] - cum[:, 0], 1 - cum[:, 1]))
    manual = -np.sum(weights * np.log(np.maximum(probs[np.arange(3), y], 1e-12)))
    manual += beta * beta / (2 * 0.01)
    c.true(abs(value - manual) < 1e-12)


def test_model_converges_without_intercept_and_rejects_invalid_fits(c: Check):
    x, y, groups = _dataset()
    model = L2CumulativeLogit().fit(x, y, groups)
    c.eq(model.beta_.shape, (6,))
    c.true(not hasattr(model, "intercept_"))
    prob = model.predict_proba(x)
    c.true(np.allclose(prob.sum(axis=1), 1.0))
    c.true(np.isfinite(model.objective_))
    c.raises(lambda: L2CumulativeLogit().fit(x * np.nan, y, groups),
             OrdinalModelError)
    c.raises(lambda: L2CumulativeLogit().fit(x, np.zeros_like(y), groups),
             OrdinalModelError, "all three grades required")


def test_group_oof_is_deterministic_disjoint_and_scaler_is_train_only(c: Check):
    x, y, groups = _dataset()
    first = group_oof_probabilities(x, y, groups, n_splits=3, return_audit=True)
    second = group_oof_probabilities(x, y, groups, n_splits=3, return_audit=True)
    c.true(np.array_equal(first.probabilities, second.probabilities))
    c.eq(len(first.folds), 3)
    for fold in first.folds:
        c.true(set(fold.train_groups).isdisjoint(fold.test_groups))
        train = x[np.isin(groups, fold.train_groups)]
        c.true(np.allclose(fold.scaler_mean, train.mean(axis=0)))
    c.eq(first.fit_count, 3)
    c.eq(first.prediction_count, len(y))


def test_weighted_metrics_keep_anchors_and_give_each_subject_total_weight_one(c: Check):
    y = np.asarray([0, 0, 1, 2, 2, 2])
    pred = np.asarray([0, 1, 1, 2, 1, 2])
    groups = np.asarray(["a", "a", "b", "c", "c", "c"])
    weights = group_total_one_weights(groups)
    c.true(np.allclose([weights[groups == g].sum() for g in ("a", "b", "c")], 1.0))
    metrics = weighted_ordinal_metrics(y, pred, groups)
    c.eq(set(metrics), {"weighted_qwk", "weighted_balanced_accuracy",
                        "macro_grade_mae"})
    c.true(abs(metrics["weighted_balanced_accuracy"] - ((0.5 + 1.0 + 2 / 3) / 3)) < 1e-12)
    c.true(abs(metrics["macro_grade_mae"] - ((0.5 + 0.0 + 1 / 3) / 3)) < 1e-12)


def test_subject_cluster_bootstrap_is_fixed_and_requires_all_grades(c: Check):
    x, y, groups = _dataset()
    pred = np.clip(np.rint(x[:, 0]), 0, 2).astype(int)
    first = subject_cluster_bootstrap(y, pred, groups, repeats=40, attempt_limit=1000)
    second = subject_cluster_bootstrap(y, pred, groups, repeats=40, attempt_limit=1000)
    c.eq(first, second)
    c.eq(first["valid_repeats"], 40)
    c.true(first["attempts"] >= 40)
    c.raises(lambda: subject_cluster_bootstrap(
        np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 1, 1]),
        np.asarray(["a", "a", "b", "b"]), repeats=2, attempt_limit=3),
        OrdinalModelError)


if __name__ == "__main__":
    run_all(__name__, globals())
