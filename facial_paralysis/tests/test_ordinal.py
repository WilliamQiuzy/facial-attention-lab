"""Validation tests for src/models/ordinal.py.

Run with the project's torch env (no pytest dependency):
    KMP_DUPLICATE_LIB_OK=TRUE \
    /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python tests/test_ordinal.py

Covers the contracts asserted in docs/model_design.md §9:
  threshold monotonicity, rank-consistency, class-probability simplex,
  loss correctness, gradient flow, shape/range validation, determinism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ordinal import (  # noqa: E402
    OrderedThresholds,
    OrdinalThresholdHead,
    class_probs,
    cum_probs,
    expected_grade,
    ordinal_loss,
    predict_grade,
)

from _testlib import Check  # noqa: E402


def test_thresholds_monotone_all_K(c: Check):
    """Thresholds are non-decreasing for K=2..8, even after random perturbation."""
    torch.manual_seed(0)
    for K in range(2, 9):
        ot = OrderedThresholds(K)
        # perturb params to simulate a mid-training state
        with torch.no_grad():
            ot.first.add_(torch.randn(()) * 2)
            if ot.gaps is not None:
                ot.gaps.add_(torch.randn_like(ot.gaps) * 2)
        th = ot.thresholds()
        c.eq(th.shape, (K - 1,), f"K={K}: threshold count")
        if K > 2:
            diffs = th[1:] - th[:-1]
            c.true(bool((diffs >= -1e-6).all()), f"K={K}: thresholds non-decreasing, diffs={diffs.tolist()}")


def test_cum_logits_shape_and_severity_monotone(c: Check):
    """Higher severity => higher cumulative logits everywhere (more severe grade)."""
    ot = OrderedThresholds(6)
    s = torch.tensor([-3.0, 0.0, 3.0])
    cl = ot(s)
    c.eq(cl.shape, (3, 5), "cum_logits shape (B, K-1)")
    # severity is monotone in cum logits per threshold
    c.true(bool((cl[2] > cl[1]).all() and (cl[1] > cl[0]).all()), "cum logits increase with severity")


def test_predict_grade_rank_consistent(c: Check):
    """predict_grade counts exceeded thresholds; sweeping severity yields a
    non-decreasing, full-range grade trajectory with ordered thresholds."""
    ot = OrderedThresholds(6, init_spread=1.0)
    s = torch.linspace(-10, 10, 41)
    grades = predict_grade(ot(s))
    c.true(bool((grades[1:] - grades[:-1] >= 0).all()), "grade non-decreasing as severity rises")
    c.eq(int(grades.min()), 0, "min grade reachable")
    c.eq(int(grades.max()), 5, "max grade reachable")
    c.true(bool(((grades >= 0) & (grades <= 5)).all()), "grades within [0,5]")


def test_class_probs_simplex(c: Check):
    """class_probs are non-negative and sum to 1 across random severities."""
    torch.manual_seed(1)
    ot = OrderedThresholds(6)
    s = torch.randn(64) * 5
    p = class_probs(ot(s))
    c.eq(p.shape, (64, 6), "class_probs shape (B, K)")
    c.true(bool((p >= -1e-7).all()), "class_probs non-negative")
    sums = p.sum(dim=1)
    c.true(bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5)), "class_probs sum to 1")


def test_predict_vs_argmax_close(c: Check):
    """predict_grade is the distribution MEDIAN (count of thresholds exceeded) and
    is the rank-consistent rule we treat as canonical; argmax(class_probs) is the
    MODE. For ordinal cumulative-link models the two agree in the large majority
    of cases (they can differ by a grade or two only for very spread-out tail
    distributions). Both must always be valid grades. This documents that the two
    decoders are consistent, not identical."""
    torch.manual_seed(2)
    ot = OrderedThresholds(6)
    s = torch.randn(2000) * 4
    cl = ot(s)
    median = predict_grade(cl)
    mode = class_probs(cl).argmax(dim=1)
    diff = (median - mode).abs()
    c.true(bool(((median >= 0) & (median <= 5)).all()), "median decoder valid grades")
    c.true(bool(((mode >= 0) & (mode <= 5)).all()), "mode decoder valid grades")
    # Stable property: the two decoders are close. Exact-agreement rate is
    # sensitive to how flat an untrained head's distributions are, so we assert
    # the robust "within one grade" rate instead.
    c.true(float((diff <= 1).float().mean()) > 0.9, "median and mode within 1 grade in >90% of cases")


def test_cum_probs_monotone_nonincreasing(c: Check):
    """P(y>k) is non-increasing in k (ordered thresholds)."""
    torch.manual_seed(3)
    ot = OrderedThresholds(6)
    q = cum_probs(ot(torch.randn(50) * 3))
    diffs = q[:, 1:] - q[:, :-1]
    c.true(bool((diffs <= 1e-6).all()), "P(y>k) non-increasing in k")


def test_expected_grade_bounds_and_monotone(c: Check):
    ot = OrderedThresholds(6)
    s = torch.linspace(-12, 12, 25)
    eg = expected_grade(ot(s))
    c.true(bool((eg >= -1e-5).all() and (eg <= 5 + 1e-5).all()), "expected grade within [0,5]")
    c.true(bool((eg[1:] - eg[:-1] >= -1e-5).all()), "expected grade non-decreasing in severity")


def test_loss_targets_and_direction(c: Check):
    """Loss for a perfect head < loss for an adversarial head; manual target check."""
    ot = OrderedThresholds(6)
    # Two samples, true grades 0 and 5.
    s_good = torch.tensor([-20.0, 20.0])   # very confident, correct ordering
    s_bad = torch.tensor([20.0, -20.0])    # confidently wrong
    y = torch.tensor([0, 5])
    l_good = ordinal_loss(ot(s_good), y)
    l_bad = ordinal_loss(ot(s_bad), y)
    c.true(float(l_good) < float(l_bad), f"correct head lower loss ({float(l_good):.3f} < {float(l_bad):.3f})")
    c.true(float(l_good) < 0.05, "near-perfect head has near-zero loss")


def test_loss_reductions(c: Check):
    ot = OrderedThresholds(4)
    cl = ot(torch.randn(8))
    y = torch.randint(0, 4, (8,))
    per = ordinal_loss(cl, y, reduction="none")
    c.eq(per.shape, (8,), "reduction='none' is per-sample (B,)")
    mean = ordinal_loss(cl, y, reduction="mean")
    c.true(bool(torch.allclose(mean, per.mean(), atol=1e-6)), "mean == mean of per-sample")
    s = ordinal_loss(cl, y, reduction="sum")
    c.true(float(s) > 0, "sum positive")


def test_gradient_flow(c: Check):
    """Loss backprops into both severity projection and thresholds."""
    head = OrdinalThresholdHead(in_dim=16, n_classes=6)
    h = torch.randn(12, 16, requires_grad=True)
    y = torch.randint(0, 6, (12,))
    loss = ordinal_loss(head(h), y)
    loss.backward()
    c.true(head.proj.weight.grad is not None and head.proj.weight.grad.abs().sum() > 0,
           "gradient reaches severity projection")
    c.true(head.thresholds.first.grad is not None, "gradient reaches threshold 'first'")
    c.true(head.thresholds.gaps.grad is not None and head.thresholds.gaps.grad.abs().sum() > 0,
           "gradient reaches threshold gaps")
    c.true(h.grad is not None and h.grad.abs().sum() > 0, "gradient reaches input representation")


def test_can_learn_planted_signal(c: Check):
    """A linear head should recover a planted monotone signal: train a few steps,
    verify ordinal loss drops substantially and predictions correlate with truth."""
    torch.manual_seed(7)
    K, N, D = 6, 240, 8
    w_true = torch.randn(D)
    X = torch.randn(N, D)
    score = X @ w_true
    # bin score into 6 ordered grades by quantiles
    qs = torch.quantile(score, torch.linspace(0, 1, K + 1)[1:-1])
    y = torch.bucketize(score, qs)
    head = OrdinalThresholdHead(in_dim=D, n_classes=K)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    with torch.no_grad():
        l0 = float(ordinal_loss(head(X), y))
    for _ in range(300):
        opt.zero_grad()
        loss = ordinal_loss(head(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        l1 = float(ordinal_loss(head(X), y))
        pred = predict_grade(head(X))
    acc = float((pred == y).float().mean())
    c.true(l1 < 0.5 * l0, f"loss dropped ({l0:.3f} -> {l1:.3f})")
    c.true(acc > 0.6, f"recovers planted signal (acc={acc:.3f})")


def test_determinism(c: Check):
    torch.manual_seed(123)
    a = OrderedThresholds(6)(torch.ones(4) * 0.3)
    torch.manual_seed(123)
    b = OrderedThresholds(6)(torch.ones(4) * 0.3)
    c.true(bool(torch.equal(a, b)), "same seed -> identical output")


def test_input_validation(c: Check):
    ot = OrderedThresholds(6)
    c.raises(lambda: ot(torch.randn(3, 2)), ValueError, "severity must be 1-D")
    c.raises(lambda: ot(torch.randint(0, 3, (4,))), TypeError, "severity must be float")
    c.raises(lambda: ordinal_loss(torch.randn(4, 5), torch.tensor([0, 1, 2, 9])),
             ValueError, "target out of range rejected")
    c.raises(lambda: ordinal_loss(torch.randn(4, 5), torch.randn(4)),
             TypeError, "float target rejected")
    c.raises(lambda: ordinal_loss(torch.randn(4, 5), torch.tensor([0, 1, 2])),
             ValueError, "batch mismatch rejected")
    c.raises(lambda: OrderedThresholds(1), ValueError, "n_classes>=2 enforced")


def test_binary_special_case(c: Check):
    """K=2: single threshold, no gaps, behaves like logistic regression."""
    ot = OrderedThresholds(2)
    c.true(ot.gaps is None, "K=2 has no gaps param")
    cl = ot(torch.tensor([-5.0, 5.0]))
    c.eq(cl.shape, (2, 1), "K=2 cum_logits shape")
    pred = predict_grade(cl)
    c.eq(pred.tolist(), [0, 1], "K=2 prediction split at threshold")
    p = class_probs(cl)
    c.true(bool(torch.allclose(p.sum(1), torch.ones(2), atol=1e-5)), "K=2 probs sum to 1")


if __name__ == "__main__":
    from _testlib import run_all
    run_all(__name__, globals())
