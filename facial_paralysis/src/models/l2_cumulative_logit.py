"""Fixed L2 proportional-odds model for YFP regional severity transfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold

FIXED_C = 0.01
PROBABILITY_FLOOR = 1e-12
MIN_CUTPOINT_GAP = 1e-6
OPTIMIZER_OPTIONS = {
    "maxiter": 2000,
    "ftol": 1e-12,
    "gtol": 1e-8,
    "maxls": 50,
}
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_ATTEMPT_LIMIT = 100000
BOOTSTRAP_SEED = 20260805


class OrdinalModelError(RuntimeError):
    """The fixed ordinal protocol cannot produce an authenticated result."""


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, value))


def _inverse_softplus(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise OrdinalModelError("softplus target must be finite and positive")
    return float(value + np.log(-np.expm1(-value)))


def ordered_cutpoints(theta0: float, raw_gap: float) -> np.ndarray:
    theta0 = float(theta0)
    raw_gap = float(raw_gap)
    result = np.asarray(
        [theta0, theta0 + _softplus(raw_gap) + MIN_CUTPOINT_GAP],
        dtype=np.float64,
    )
    if not np.isfinite(result).all() or result[1] <= result[0]:
        raise OrdinalModelError("cut-points must be finite and strictly ordered")
    return result


def initial_parameters(n_features: int) -> np.ndarray:
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features <= 0:
        raise OrdinalModelError("n_features must be a positive integer")
    raw_gap = _inverse_softplus(1.0 - MIN_CUTPOINT_GAP)
    return np.concatenate((
        np.zeros(n_features, dtype=np.float64),
        np.asarray([-0.5, raw_gap], dtype=np.float64),
    ))


def cumulative_probabilities(
    x: np.ndarray,
    beta: np.ndarray,
    cutpoints: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    cutpoints = np.asarray(cutpoints, dtype=np.float64)
    if (x.ndim != 2 or beta.shape != (x.shape[1],)
            or cutpoints.shape != (2,) or not np.isfinite(x).all()
            or not np.isfinite(beta).all() or not np.isfinite(cutpoints).all()
            or cutpoints[1] <= cutpoints[0]):
        raise OrdinalModelError("invalid cumulative-logit inputs")
    return expit(cutpoints[None, :] - (x @ beta)[:, None])


def class_probabilities(
    x: np.ndarray,
    beta: np.ndarray,
    cutpoints: np.ndarray,
) -> np.ndarray:
    cumulative = cumulative_probabilities(x, beta, cutpoints)
    probability = np.column_stack((
        cumulative[:, 0],
        cumulative[:, 1] - cumulative[:, 0],
        1.0 - cumulative[:, 1],
    ))
    probability = np.maximum(probability, PROBABILITY_FLOOR)
    probability /= probability.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all():
        raise OrdinalModelError("class probability is nonfinite")
    return probability


def group_total_one_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise OrdinalModelError("groups must be a nonempty vector")
    weights = np.empty(len(groups), dtype=np.float64)
    for group in np.unique(groups):
        mask = groups == group
        weights[mask] = 1.0 / int(mask.sum())
    return weights


def proportional_odds_objective(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or weights.shape != (len(x),)
            or parameters.shape != (x.shape[1] + 2,)
            or not np.isfinite(x).all() or not np.isfinite(weights).all()
            or not np.isfinite(parameters).all() or np.any(weights <= 0)
            or np.any((y < 0) | (y > 2))):
        return float("inf")
    beta = parameters[:x.shape[1]]
    try:
        cutpoints = ordered_cutpoints(parameters[-2], parameters[-1])
        probability = class_probabilities(x, beta, cutpoints)
    except OrdinalModelError:
        return float("inf")
    selected = np.maximum(probability[np.arange(len(y)), y], PROBABILITY_FLOOR)
    value = -float(np.sum(weights * np.log(selected)))
    value += float(beta @ beta) / (2.0 * FIXED_C)
    return value if np.isfinite(value) else float("inf")


class L2CumulativeLogit:
    """Three-class fixed-C proportional-odds linear model (no intercept)."""

    def __init__(self) -> None:
        self.beta_: np.ndarray
        self.cutpoints_: np.ndarray
        self.objective_: float
        self.optimizer_result_: Any

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "L2CumulativeLogit":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups)
        if (x.ndim != 2 or len(x) == 0 or y.shape != (len(x),)
                or groups.shape != (len(x),) or not np.isfinite(x).all()):
            raise OrdinalModelError("fit inputs must be finite aligned matrices")
        if set(np.unique(y).tolist()) != {0, 1, 2}:
            raise OrdinalModelError("every training fold must contain all three grades")
        weights = group_total_one_weights(groups)
        result = minimize(
            proportional_odds_objective,
            initial_parameters(x.shape[1]),
            args=(x, y, weights),
            method="L-BFGS-B",
            options=dict(OPTIMIZER_OPTIONS),
        )
        objective = float(result.fun)
        if not result.success or not np.isfinite(objective) or not np.isfinite(result.x).all():
            raise OrdinalModelError(
                f"L-BFGS-B failed to converge: status={result.status}, message={result.message}"
            )
        try:
            hessian_inverse = np.asarray(result.hess_inv.todense(), dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise OrdinalModelError("optimizer did not provide a finite Hessian inverse") from exc
        if hessian_inverse.shape != (x.shape[1] + 2, x.shape[1] + 2) or not np.isfinite(hessian_inverse).all():
            raise OrdinalModelError("optimizer Hessian inverse is nonfinite")
        self.beta_ = np.asarray(result.x[:-2], dtype=np.float64)
        self.cutpoints_ = ordered_cutpoints(result.x[-2], result.x[-1])
        self.objective_ = objective
        self.optimizer_result_ = result
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "beta_"):
            raise OrdinalModelError("model has not been fitted")
        return class_probabilities(np.asarray(x, dtype=np.float64),
                                   self.beta_, self.cutpoints_)


@dataclass(frozen=True)
class OOFAuditFold:
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


@dataclass(frozen=True)
class OOFResult:
    probabilities: np.ndarray
    folds: tuple[OOFAuditFold, ...]
    fit_count: int
    prediction_count: int


def group_oof_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    return_audit: bool = False,
) -> OOFResult | np.ndarray:
    """Deterministic group-disjoint OOF with train-fold-only standardization."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if (x.ndim != 2 or y.shape != (len(x),) or groups.shape != (len(x),)
            or not np.isfinite(x).all() or set(np.unique(y).tolist()) != {0, 1, 2}
            or n_splits < 2 or n_splits > len(np.unique(groups))):
        raise OrdinalModelError("invalid group OOF inputs")
    probability = np.full((len(x), 3), np.nan, dtype=np.float64)
    folds: list[OOFAuditFold] = []
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, test_index in splitter.split(x, y, groups):
        train_groups = tuple(sorted(str(value) for value in np.unique(groups[train_index])))
        test_groups = tuple(sorted(str(value) for value in np.unique(groups[test_index])))
        if set(train_groups) & set(test_groups):
            raise OrdinalModelError("group leakage in OOF split")
        mean = x[train_index].mean(axis=0)
        scale = x[train_index].std(axis=0)
        scale[scale == 0.0] = 1.0
        train_x = (x[train_index] - mean) / scale
        test_x = (x[test_index] - mean) / scale
        model = L2CumulativeLogit().fit(train_x, y[train_index], groups[train_index])
        probability[test_index] = model.predict_proba(test_x)
        folds.append(OOFAuditFold(train_groups, test_groups, mean.copy(), scale.copy()))
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0):
        raise OrdinalModelError("OOF prediction coverage is incomplete")
    result = OOFResult(probability, tuple(folds), len(folds), len(x))
    return result if return_audit else result.probabilities


def _weighted_confusion(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    confusion = np.zeros((3, 3), dtype=np.float64)
    for true_value, pred_value, weight in zip(y, pred, weights):
        confusion[int(true_value), int(pred_value)] += float(weight)
    return confusion


def weighted_ordinal_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    groups = np.asarray(groups)
    if (y.shape != pred.shape or y.shape != groups.shape or y.ndim != 1
            or set(np.unique(y).tolist()) != {0, 1, 2}
            or np.any((pred < 0) | (pred > 2))):
        raise OrdinalModelError("metrics require aligned 0/1/2 anchor vectors")
    weights = group_total_one_weights(groups)
    confusion = _weighted_confusion(y, pred, weights)
    recalls = np.divide(np.diag(confusion), confusion.sum(axis=1),
                        out=np.zeros(3), where=confusion.sum(axis=1) > 0)
    grade_mae = []
    for grade in range(3):
        mask = y == grade
        grade_mae.append(float(np.average(np.abs(pred[mask] - grade),
                                          weights=weights[mask])))
    total = float(confusion.sum())
    observed = confusion / total
    true_marginal = observed.sum(axis=1)
    pred_marginal = observed.sum(axis=0)
    expected = np.outer(true_marginal, pred_marginal)
    penalty = np.asarray([[((i - j) / 2.0) ** 2 for j in range(3)] for i in range(3)])
    denominator = float(np.sum(penalty * expected))
    if denominator <= 0 or not np.isfinite(denominator):
        raise OrdinalModelError("weighted QWK denominator is degenerate")
    qwk = 1.0 - float(np.sum(penalty * observed)) / denominator
    metrics = {
        "weighted_qwk": qwk,
        "weighted_balanced_accuracy": float(np.mean(recalls)),
        "macro_grade_mae": float(np.mean(grade_mae)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise OrdinalModelError("ordinal metrics are nonfinite")
    return metrics


def subject_cluster_bootstrap(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    attempt_limit: int = BOOTSTRAP_ATTEMPT_LIMIT,
) -> dict[str, Any]:
    """Fixed-seed subject-cluster percentile intervals for one target."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    groups = np.asarray(groups)
    if repeats <= 0 or attempt_limit < repeats or y.shape != pred.shape or y.shape != groups.shape:
        raise OrdinalModelError("invalid subject-cluster bootstrap inputs")
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {
        "weighted_qwk": [],
        "weighted_balanced_accuracy": [],
        "macro_grade_mae": [],
    }
    attempts = 0
    while len(samples["weighted_qwk"]) < repeats and attempts < attempt_limit:
        attempts += 1
        draw = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        y_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        bootstrap_groups: list[np.ndarray] = []
        for occurrence, group in enumerate(draw):
            mask = groups == group
            y_parts.append(y[mask])
            pred_parts.append(pred[mask])
            bootstrap_groups.append(np.asarray(
                [f"draw_{occurrence}"] * int(mask.sum()), dtype=object))
        draw_y = np.concatenate(y_parts)
        if set(np.unique(draw_y).tolist()) != {0, 1, 2}:
            continue
        metrics = weighted_ordinal_metrics(
            draw_y, np.concatenate(pred_parts), np.concatenate(bootstrap_groups))
        for key in samples:
            samples[key].append(metrics[key])
    if len(samples["weighted_qwk"]) != repeats:
        raise OrdinalModelError(
            f"only {len(samples['weighted_qwk'])} valid all-grade draws in {attempts} attempts"
        )
    intervals = {
        key: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for key, values in samples.items()
    }
    return {
        "seed": BOOTSTRAP_SEED,
        "valid_repeats": repeats,
        "attempts": attempts,
        "attempt_limit": attempt_limit,
        "intervals_95_percentile": intervals,
    }


__all__ = [
    "BOOTSTRAP_ATTEMPT_LIMIT", "BOOTSTRAP_REPEATS", "BOOTSTRAP_SEED",
    "FIXED_C", "OPTIMIZER_OPTIONS", "PROBABILITY_FLOOR",
    "L2CumulativeLogit", "OOFAuditFold", "OOFResult", "OrdinalModelError",
    "class_probabilities", "cumulative_probabilities",
    "group_oof_probabilities", "group_total_one_weights",
    "initial_parameters", "ordered_cutpoints", "proportional_odds_objective",
    "subject_cluster_bootstrap", "weighted_ordinal_metrics",
]
