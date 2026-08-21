"""Ordinal cut-point head and loss for facial-paralysis severity grading.

This implements the latent-severity / ordered-threshold ("cumulative link")
model described in `docs/model_design.md` §4–5. It is the building block that
lets datasets on different label scales (binary palsy, 3-class coarse, HB I–VI,
region intensity) all attach to a *single* scalar severity `s`:

    P(y > k) = sigmoid(s - theta_k),   k = 0 .. K-2

with thresholds theta_1 <= theta_2 <= ... <= theta_{K-1} kept monotone *by
construction* (cumulative softplus gaps), so rank-consistency can never be
violated during training. This is the CORAL-style consistent-rank ordinal model
(Cao et al. 2020), reparameterized so the thresholds — not arbitrary biases —
are the learned objects, which is what makes one severity sharable across heads.

The module is deliberately self-contained (only torch) so it can be unit-tested
and reused without pulling in the rest of the project.

Conventions:
  * Labels are 0-indexed grades in {0, .., K-1}. For HB, grade 0 == clinical
    Grade I (normal), grade 5 == Grade VI (total paralysis).
  * "cumulative logits" always have shape (B, K-1) and equal `s - theta_k`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Validation helpers (kept terse but loud — see docs/model_design.md §9)
# ----------------------------------------------------------------------
def _check_severity(s: torch.Tensor) -> None:
    if s.dim() != 1:
        raise ValueError(f"severity must be 1-D (B,), got shape {tuple(s.shape)}")
    if not torch.is_floating_point(s):
        raise TypeError(f"severity must be a float tensor, got dtype {s.dtype}")


def _check_cum_logits(cum_logits: torch.Tensor, n_classes: int | None = None) -> None:
    if cum_logits.dim() != 2:
        raise ValueError(
            f"cumulative logits must be 2-D (B, K-1), got shape {tuple(cum_logits.shape)}"
        )
    if cum_logits.shape[1] < 1:
        raise ValueError("cumulative logits need at least 1 threshold (n_classes >= 2)")
    if n_classes is not None and cum_logits.shape[1] != n_classes - 1:
        raise ValueError(
            f"expected K-1={n_classes - 1} thresholds, got {cum_logits.shape[1]}"
        )


def _check_targets(target: torch.Tensor, n_classes: int) -> None:
    if target.dim() != 1:
        raise ValueError(f"target must be 1-D (B,), got shape {tuple(target.shape)}")
    if target.dtype not in (torch.int64, torch.int32, torch.int16, torch.long):
        raise TypeError(f"target must be an integer/long tensor, got dtype {target.dtype}")
    if target.numel() > 0:
        lo, hi = int(target.min()), int(target.max())
        if lo < 0 or hi > n_classes - 1:
            raise ValueError(
                f"target values must be in [0, {n_classes - 1}], got [{lo}, {hi}]"
            )


# ----------------------------------------------------------------------
# Ordered thresholds on a scalar severity
# ----------------------------------------------------------------------
class OrderedThresholds(nn.Module):
    """K-1 monotonically increasing cut-points on a scalar latent severity.

    Parameterization guarantees theta_1 <= theta_2 <= ... <= theta_{K-1}:

        theta = first + concat([0, cumsum(softplus(gaps))])

    where `first` is free (R) and `gaps` (length K-2) map through softplus to
    non-negative increments. For K == 2 there is a single free threshold and no
    gaps. Because the ordering holds for *any* parameter values, the optimizer
    can never produce a rank-inconsistent head.

    forward(severity) -> cumulative logits  s - theta_k,  shape (B, K-1).
    """

    def __init__(self, n_classes: int, init_spread: float = 1.0):
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes = int(n_classes)
        n_thresh = n_classes - 1
        # Initialize thresholds spread symmetrically around 0 so an untrained
        # head with severity ~0 starts near a uniform-ish class distribution.
        # first ~ -((K-2)/2)*init_spread ; gaps ~ init_spread each.
        self.first = nn.Parameter(torch.tensor(-0.5 * (n_thresh - 1) * init_spread))
        if n_thresh > 1:
            # softplus(raw) = init_spread  ->  raw = log(exp(init_spread) - 1)
            raw = torch.log(torch.expm1(torch.tensor(float(init_spread)))).clamp_min(-10.0)
            self.gaps = nn.Parameter(raw.repeat(n_thresh - 1))
        else:
            # No gaps to learn; register a 0-length buffer for a uniform code path.
            self.register_parameter("gaps", None)

    def thresholds(self) -> torch.Tensor:
        """Return the ordered thresholds, shape (K-1,)."""
        if self.gaps is None:
            return self.first.reshape(1)
        increments = F.softplus(self.gaps)                       # (K-2,) >= 0
        cumulative = torch.cumsum(increments, dim=0)             # (K-2,)
        zero = torch.zeros(1, device=cumulative.device, dtype=cumulative.dtype)
        offsets = torch.cat([zero, cumulative], dim=0)           # (K-1,)
        return self.first + offsets

    def forward(self, severity: torch.Tensor) -> torch.Tensor:
        _check_severity(severity)
        theta = self.thresholds()                                # (K-1,)
        return severity.unsqueeze(1) - theta.unsqueeze(0)        # (B, K-1)


# ----------------------------------------------------------------------
# A head that owns its severity projection
# ----------------------------------------------------------------------
class OrdinalThresholdHead(nn.Module):
    """Project a representation to a scalar severity, then apply ordered
    thresholds. Use this for tasks that should have their *own* severity axis
    (e.g. YFP per-region intensity). For tasks that share a global severity,
    feed an externally-computed severity straight to `OrderedThresholds`.

    forward(h) -> cumulative logits (B, K-1).
    forward_severity(h) -> scalar severity (B,) (exposed for inspection/coupling).
    """

    def __init__(self, in_dim: int, n_classes: int, init_spread: float = 1.0):
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        self.proj = nn.Linear(in_dim, 1)
        self.thresholds = OrderedThresholds(n_classes, init_spread=init_spread)
        self.n_classes = n_classes

    def forward_severity(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"expected (B, in_dim), got shape {tuple(h.shape)}")
        return self.proj(h).squeeze(1)                           # (B,)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.thresholds(self.forward_severity(h))


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def ordinal_loss(
    cum_logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Consistent-rank (CORAL-style) ordinal loss.

    For label y, the K-1 binary sub-tasks have targets t_k = 1[y > k],
    k = 0..K-2. Loss is BCE-with-logits over the cumulative logits.

    Args:
        cum_logits: (B, K-1) = s - theta_k.
        target:     (B,) long, values in [0, K-1].
        reduction:  'mean' | 'sum' | 'none'. 'none' returns per-sample loss (B,)
                    (mean over thresholds within each sample).
    """
    _check_cum_logits(cum_logits)
    n_classes = cum_logits.shape[1] + 1
    _check_targets(target, n_classes)
    if cum_logits.shape[0] != target.shape[0]:
        raise ValueError(
            f"batch mismatch: cum_logits {cum_logits.shape[0]} vs target {target.shape[0]}"
        )

    device = cum_logits.device
    k = torch.arange(n_classes - 1, device=device)               # (K-1,)
    # t[b, k] = 1 if target[b] > k else 0
    binary_targets = (target.unsqueeze(1) > k.unsqueeze(0)).to(cum_logits.dtype)

    per_elem = F.binary_cross_entropy_with_logits(
        cum_logits, binary_targets, reduction="none"
    )                                                            # (B, K-1)
    if reduction == "none":
        return per_elem.mean(dim=1)                              # (B,)
    if reduction == "sum":
        return per_elem.sum()
    if reduction == "mean":
        return per_elem.mean()
    raise ValueError(f"unknown reduction: {reduction}")


# ----------------------------------------------------------------------
# Decoding: probabilities, predicted grade, expected grade
# ----------------------------------------------------------------------
def cum_probs(cum_logits: torch.Tensor) -> torch.Tensor:
    """P(y > k) for k = 0..K-2, shape (B, K-1). Monotone non-increasing in k
    whenever thresholds are ordered (guaranteed by OrderedThresholds)."""
    _check_cum_logits(cum_logits)
    return torch.sigmoid(cum_logits)


def class_probs(cum_logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative logits to a proper class distribution, shape (B, K).

        P(y=0)   = 1 - P(y>0)
        P(y=j)   = P(y>j-1) - P(y>j)      0 < j < K-1
        P(y=K-1) = P(y>K-2)

    With ordered thresholds these are all >= 0 and sum to 1. We clamp tiny
    negative values that can appear if a caller feeds *unordered* logits, so the
    output is always a valid simplex; ordered heads never trigger the clamp.
    """
    q = cum_probs(cum_logits)                                    # (B, K-1) = P(y>k)
    B = q.shape[0]
    ones = torch.ones(B, 1, device=q.device, dtype=q.dtype)
    zeros = torch.zeros(B, 1, device=q.device, dtype=q.dtype)
    upper = torch.cat([ones, q], dim=1)                          # P(y > k-1), incl P(y>-1)=1
    lower = torch.cat([q, zeros], dim=1)                         # P(y > k),   incl P(y>K-1)=0
    probs = (upper - lower).clamp_min(0.0)                       # (B, K)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs


def predict_grade(cum_logits: torch.Tensor) -> torch.Tensor:
    """Rank-consistent predicted grade, shape (B,) long.

    grade = number of thresholds the severity exceeds = sum_k 1[P(y>k) > 0.5].
    With ordered thresholds this equals argmax over class_probs in the common
    case and is always a valid grade in [0, K-1]."""
    _check_cum_logits(cum_logits)
    return (cum_logits > 0).sum(dim=1).long()                    # (B,)


def expected_grade(cum_logits: torch.Tensor) -> torch.Tensor:
    """Continuous expected grade E[y] = sum_j j * P(y=j), shape (B,). Useful as a
    smooth readout and for MAE-style monitoring."""
    p = class_probs(cum_logits)                                  # (B, K)
    j = torch.arange(p.shape[1], device=p.device, dtype=p.dtype) # (K,)
    return (p * j.unsqueeze(0)).sum(dim=1)                        # (B,)
