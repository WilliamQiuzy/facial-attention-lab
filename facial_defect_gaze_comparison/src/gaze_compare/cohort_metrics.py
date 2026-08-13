from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class IndependentEquivalenceResult:
    mean_difference: float
    lower: float
    upper: float
    margin: float
    confidence: float
    outcome: str
    n_webcam: int
    n_professional: int
    hedges_g: float


@dataclass(frozen=True)
class ClassifierAUCResult:
    auc: float
    lower: float
    upper: float
    n: int
    n_splits: int
    n_repeats: int


def _finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain at least two finite values")
    return array


def standardized_mean_difference(
    webcam: Iterable[float], professional: Iterable[float]
) -> float:
    """Return webcam minus professional mean, scaled by pooled sample SD."""

    first = _finite_vector(webcam, name="webcam")
    second = _finite_vector(professional, name="professional")
    numerator = float(first.mean() - second.mean())
    pooled_variance = (
        (len(first) - 1) * np.var(first, ddof=1)
        + (len(second) - 1) * np.var(second, ddof=1)
    ) / (len(first) + len(second) - 2)
    if pooled_variance <= 0:
        return 0.0 if numerator == 0 else float(np.sign(numerator) * np.inf)
    return float(numerator / np.sqrt(pooled_variance))


def _hedges_g(webcam: np.ndarray, professional: np.ndarray) -> float:
    effect = standardized_mean_difference(webcam, professional)
    degrees_freedom = len(webcam) + len(professional) - 2
    correction = 1 - 3 / (4 * degrees_freedom - 1) if degrees_freedom > 1 else 1.0
    return float(effect * correction)


def independent_equivalence(
    webcam: Iterable[float],
    professional: Iterable[float],
    *,
    margin: float,
    confidence: float = 0.90,
) -> IndependentEquivalenceResult:
    """Welch interval with a transparent three-way equivalence interpretation.

    A 90% interval fully inside +/-margin supports equivalence by TOST. An interval
    fully beyond either margin is called meaningfully different; all overlap cases
    remain inconclusive. The margin must be chosen before inspecting real outcomes.
    """

    first = _finite_vector(webcam, name="webcam")
    second = _finite_vector(professional, name="professional")
    if margin <= 0 or not 0 < confidence < 1:
        raise ValueError("margin must be positive and confidence must be between 0 and 1")

    difference = float(first.mean() - second.mean())
    first_term = float(np.var(first, ddof=1) / len(first))
    second_term = float(np.var(second, ddof=1) / len(second))
    standard_error = np.sqrt(first_term + second_term)
    if standard_error == 0:
        lower = upper = difference
    else:
        numerator = (first_term + second_term) ** 2
        denominator = first_term**2 / (len(first) - 1) + second_term**2 / (len(second) - 1)
        degrees_freedom = numerator / denominator
        critical = float(stats.t.ppf(1 - (1 - confidence) / 2, degrees_freedom))
        lower = float(difference - critical * standard_error)
        upper = float(difference + critical * standard_error)

    if lower > -margin and upper < margin:
        outcome = "similar_within_margin"
    elif lower > margin or upper < -margin:
        outcome = "meaningfully_different"
    else:
        outcome = "inconclusive"
    return IndependentEquivalenceResult(
        mean_difference=difference,
        lower=lower,
        upper=upper,
        margin=float(margin),
        confidence=float(confidence),
        outcome=outcome,
        n_webcam=len(first),
        n_professional=len(second),
        hedges_g=_hedges_g(first, second),
    )


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise ValueError("AUC requires both classes")
    ranks = stats.rankdata(scores, method="average")
    rank_sum = float(ranks[positive].sum())
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coefficients
        penalty = 0.5 * float(np.square(coefficients[1:]).sum()) / len(features)
        loss = float(np.mean(np.logaddexp(0, logits) - labels * logits) + penalty)
        probabilities = special.expit(logits)
        gradient = design.T @ (probabilities - labels) / len(features)
        gradient[1:] += coefficients[1:] / len(features)
        return loss, gradient

    fitted = optimize.minimize(
        objective,
        np.zeros(design.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300},
    )
    if not fitted.success:
        raise RuntimeError(f"domain classifier did not converge: {fitted.message}")
    return fitted.x


def repeated_cross_validated_auc(
    features: np.ndarray,
    labels: Iterable[int],
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_boot: int = 500,
    seed: int = 20260813,
) -> ClassifierAUCResult:
    """Repeated stratified CV AUC with participant bootstrap uncertainty."""

    matrix = np.asarray(features, dtype=float)
    target = np.asarray(list(labels), dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != len(target) or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite 2D matrix aligned with labels")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("labels must contain exactly 0 and 1")
    if n_splits < 2 or n_repeats < 1 or n_boot < 100:
        raise ValueError("use at least 2 folds, 1 repeat, and 100 bootstrap replicates")
    class_indices = {label: np.flatnonzero(target == label) for label in (0, 1)}
    if min(map(len, class_indices.values())) < n_splits:
        raise ValueError("each class must contain at least n_splits participants")

    rng = np.random.default_rng(seed)
    score_sum = np.zeros(len(target), dtype=float)
    score_count = np.zeros(len(target), dtype=int)
    for _ in range(n_repeats):
        folds: dict[int, list[np.ndarray]] = {}
        for label in (0, 1):
            folds[label] = list(np.array_split(rng.permutation(class_indices[label]), n_splits))
        for fold_index in range(n_splits):
            test_indices = np.concatenate([folds[label][fold_index] for label in (0, 1)])
            train_mask = np.ones(len(target), dtype=bool)
            train_mask[test_indices] = False
            train_features = matrix[train_mask]
            means = train_features.mean(axis=0)
            scales = train_features.std(axis=0, ddof=0)
            scales[scales == 0] = 1.0
            standardized_train = (train_features - means) / scales
            coefficients = _fit_logistic(standardized_train, target[train_mask])
            standardized_test = (matrix[test_indices] - means) / scales
            scores = special.expit(
                coefficients[0] + standardized_test @ coefficients[1:]
            )
            score_sum[test_indices] += scores
            score_count[test_indices] += 1

    if (score_count == 0).any():
        raise RuntimeError("cross-validation failed to score every participant")
    out_of_fold_scores = score_sum / score_count
    auc = float(_binary_auc(target, out_of_fold_scores))

    bootstrap = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = np.concatenate(
            [rng.choice(class_indices[label], size=len(class_indices[label]), replace=True) for label in (0, 1)]
        )
        bootstrap[index] = _binary_auc(target[sampled], out_of_fold_scores[sampled])
    return ClassifierAUCResult(
        auc=auc,
        lower=float(np.quantile(bootstrap, 0.025)),
        upper=float(np.quantile(bootstrap, 0.975)),
        n=len(target),
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
