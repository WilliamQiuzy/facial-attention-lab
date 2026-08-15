"""Nested participant-level contracts for Universal Phenotype Mixture v3."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.universal_phenotype_v3 import MODEL_VARIANTS


SOURCES = ("palsynet", "neuroface", "meei")
PHENOTYPES = ("healthy", "palsy", "als", "post_stroke")
OUTER_FOLDS = 6
INNER_FOLDS = 5
_GROUP_ID = re.compile(r"grp_[0-9a-f]{64}")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    variant: str | None
    width: int
    dropout: float
    learning_rate: float
    alignment_weight: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or re.fullmatch(r"[a-z0-9_]+", self.name) is None
            or self.family not in {"locked_landmark", "neural_mixture"}
            or isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or not math.isfinite(float(self.dropout))
            or not math.isfinite(float(self.learning_rate))
            or self.alignment_weight not in {0.0, 0.05}
        ):
            raise ValueError("universal candidate specification is invalid")
        if self.family == "locked_landmark":
            if (
                self.variant is not None or self.width != 0
                or self.dropout != 0.0 or self.learning_rate != 0.0
                or self.alignment_weight != 0.0
            ):
                raise ValueError("locked landmark candidate cannot carry neural controls")
        elif (
            self.variant not in MODEL_VARIANTS
            or self.width not in {32, 64}
            or self.dropout not in {0.1, 0.3}
            or self.learning_rate not in {0.001, 0.003}
        ):
            raise ValueError("neural candidate differs from the bounded search space")


FROZEN_CANDIDATES = (
    CandidateSpec(
        "locked_landmark110", "locked_landmark", None, 0, 0.0, 0.0, 0.0
    ),
    CandidateSpec(
        "linear_expert_w32", "neural_mixture", "linear_expert_mixture",
        32, 0.1, 0.001, 0.0,
    ),
    CandidateSpec(
        "residual_mil_w32", "neural_mixture", "residual_mil",
        32, 0.1, 0.001, 0.0,
    ),
    CandidateSpec(
        "residual_mil_w64", "neural_mixture", "residual_mil",
        64, 0.3, 0.003, 0.05,
    ),
    CandidateSpec(
        "hybrid_tcn_w32", "neural_mixture", "hybrid_tcn_mil",
        32, 0.1, 0.001, 0.0,
    ),
    CandidateSpec(
        "hybrid_tcn_w64", "neural_mixture", "hybrid_tcn_mil",
        64, 0.3, 0.003, 0.05,
    ),
    CandidateSpec(
        "hybrid_set_w32", "neural_mixture", "hybrid_set_transformer",
        32, 0.1, 0.001, 0.0,
    ),
    CandidateSpec(
        "hybrid_set_w64", "neural_mixture", "hybrid_set_transformer",
        64, 0.3, 0.003, 0.05,
    ),
)


def _immutable(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class NestedFold:
    outer_train: np.ndarray
    outer_held: np.ndarray
    inner_folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def _validated_metadata(
    labels: np.ndarray,
    group_ids: Sequence[str],
    sources: Sequence[str],
    phenotypes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    labels = np.asarray(labels)
    group_ids = tuple(group_ids)
    sources = tuple(sources)
    phenotypes = tuple(phenotypes)
    count = len(group_ids)
    if (
        labels.shape != (count,)
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or len(sources) != count
        or len(phenotypes) != count
        or len(set(group_ids)) != count
        or any(not isinstance(group, str) or _GROUP_ID.fullmatch(group) is None
               for group in group_ids)
        or any(source not in SOURCES for source in sources)
        or any(phenotype not in PHENOTYPES for phenotype in phenotypes)
        or any((phenotype == "healthy") != (int(label) == 0)
               for phenotype, label in zip(phenotypes, labels))
    ):
        raise ValueError("nested folds require unique aligned participant metadata")
    return labels, group_ids, sources, phenotypes


def _stratified_assignment(
    indices: np.ndarray,
    *,
    fold_count: int,
    group_ids: tuple[str, ...],
    sources: tuple[str, ...],
    phenotypes: tuple[str, ...],
) -> np.ndarray:
    assignment = np.full(len(group_ids), -1, dtype=np.int64)
    cells = sorted({(sources[index], phenotypes[index]) for index in indices})
    for cell in cells:
        selected = sorted(
            (int(index) for index in indices
             if (sources[int(index)], phenotypes[int(index)]) == cell),
            key=lambda index: group_ids[index],
        )
        if len(selected) < fold_count:
            raise ValueError("every source-phenotype cell must cover every fold")
        for position, index in enumerate(selected):
            assignment[index] = position % fold_count
    if np.any(assignment[indices] < 0):
        raise ValueError("a participant escaped source-phenotype stratification")
    return assignment


def nested_source_phenotype_folds(
    labels: np.ndarray,
    group_ids: Sequence[str],
    sources: Sequence[str],
    phenotypes: Sequence[str],
) -> tuple[NestedFold, ...]:
    """Create deterministic 6x5 nested folds with global participant indices."""
    labels, group_ids, sources, phenotypes = _validated_metadata(
        labels, group_ids, sources, phenotypes
    )
    all_indices = np.arange(len(group_ids), dtype=np.int64)
    outer_assignment = _stratified_assignment(
        all_indices, fold_count=OUTER_FOLDS, group_ids=group_ids,
        sources=sources, phenotypes=phenotypes,
    )
    folds = []
    for outer in range(OUTER_FOLDS):
        outer_held = all_indices[outer_assignment == outer]
        outer_train = all_indices[outer_assignment != outer]
        inner_assignment = _stratified_assignment(
            outer_train, fold_count=INNER_FOLDS, group_ids=group_ids,
            sources=sources, phenotypes=phenotypes,
        )
        inner = tuple(
            (
                _immutable(outer_train[inner_assignment[outer_train] != fold]),
                _immutable(outer_train[inner_assignment[outer_train] == fold]),
            )
            for fold in range(INNER_FOLDS)
        )
        folds.append(NestedFold(
            outer_train=_immutable(outer_train),
            outer_held=_immutable(outer_held),
            inner_folds=inner,
        ))
    return tuple(folds)


def summarize_source_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sources: Sequence[str],
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    sources = tuple(sources)
    if (
        labels.shape != probabilities.shape
        or labels.ndim != 1
        or len(sources) != len(labels)
        or set(sources) != set(SOURCES)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("source metrics require complete three-source probabilities")
    result = {"overall": binary_metrics(labels, probabilities)}
    for source in SOURCES:
        selected = np.asarray([value == source for value in sources], dtype=bool)
        result[source] = binary_metrics(labels[selected], probabilities[selected])
    return result


def select_inner_global_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Choose one source-blind threshold from inner participant OOF scores."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != probabilities.shape
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or set(labels.tolist()) != {0, 1}
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("inner threshold selection requires finite binary OOF scores")
    unique = np.unique(probabilities)
    candidates = np.unique(np.concatenate((
        np.asarray((0.0, 1.0), dtype=np.float64),
        0.5 * (unique[:-1] + unique[1:]),
    )))
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    best: tuple[float, float, float] | None = None
    for threshold in candidates.tolist():
        predicted = probabilities >= threshold
        sensitivity = float(np.sum(predicted & (labels == 1)) / positive_count)
        specificity = float(np.sum(~predicted & (labels == 0)) / negative_count)
        balanced = 0.5 * (sensitivity + specificity)
        key = (balanced, sensitivity, -float(threshold))
        if best is None or key > best:
            best = key
    assert best is not None
    return {
        "threshold": float(-best[2]),
        "balanced_accuracy": float(best[0]),
    }


def select_worst_source_candidate(
    summaries: dict[str, dict[str, float]],
) -> str:
    """Select by minimum source AUC, minimum BA, maximum Brier, then order."""
    names = tuple(candidate.name for candidate in FROZEN_CANDIDATES)
    if set(summaries) != set(names):
        raise ValueError("selection requires the exact frozen candidate registry")
    ranking = []
    for order, name in enumerate(names):
        row = summaries[name]
        required = (
            "worst_source_auroc", "worst_source_balanced_accuracy",
            "worst_source_brier",
        )
        if any(key not in row for key in required):
            raise ValueError("candidate summary lacks robust selection metrics")
        auroc, balanced, brier = (float(row[key]) for key in required)
        if (
            not all(math.isfinite(value) for value in (auroc, balanced, brier))
            or not 0.0 <= auroc <= 1.0
            or not 0.0 <= balanced <= 1.0
            or not 0.0 <= brier <= 1.0
        ):
            raise ValueError("robust selection metrics are invalid")
        ranking.append((auroc, balanced, -brier, -order, name))
    return max(ranking)[-1]


__all__ = (
    "CandidateSpec",
    "FROZEN_CANDIDATES",
    "INNER_FOLDS",
    "NestedFold",
    "OUTER_FOLDS",
    "nested_source_phenotype_folds",
    "select_inner_global_threshold",
    "select_worst_source_candidate",
    "summarize_source_metrics",
)
