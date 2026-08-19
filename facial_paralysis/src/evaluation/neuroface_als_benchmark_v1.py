"""Participant-level NeuroFace ALS-versus-healthy benchmark protocols."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


PAPER_ACCURACY = 0.91
PAPER_AUROC = 0.97
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PENALTIES = ("l1", "l2")
CORRELATION_THRESHOLD = 0.7
RANDOM_STATE = 42


def _frozen(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return result.reshape(contiguous.shape)


@dataclass(frozen=True, order=True)
class Candidate:
    representation: str
    penalty: str
    c: float

    def __post_init__(self):
        if (not isinstance(self.representation, str) or not self.representation
                or self.penalty not in PENALTIES or self.c not in C_GRID):
            raise ValueError("candidate differs from the frozen paper search space")


@dataclass(frozen=True)
class FoldFeatures:
    train: np.ndarray
    test: np.ndarray
    kept_indices: np.ndarray
    train_mean: np.ndarray
    train_scale: np.ndarray


@dataclass(frozen=True)
class FixedEvaluation:
    candidate: Candidate
    probabilities: np.ndarray
    metrics: dict[str, float]
    n_participants: int
    selection_protocol: str


@dataclass(frozen=True)
class ThresholdCandidateSelection:
    candidate: Candidate
    probabilities: np.ndarray
    predictions: np.ndarray
    threshold: float
    metrics: dict[str, float]
    n_participants: int
    selection_protocol: str


@dataclass(frozen=True)
class NestedEvaluation:
    probabilities: np.ndarray
    metrics: dict[str, float]
    outer_candidates: tuple[Candidate, ...]
    n_participants: int
    selection_protocol: str


@dataclass(frozen=True)
class ThresholdedNestedEvaluation:
    probabilities: np.ndarray
    predictions: np.ndarray
    outer_thresholds: np.ndarray
    metrics: dict[str, float]
    outer_candidates: tuple[Candidate, ...]
    n_participants: int
    selection_protocol: str


@dataclass(frozen=True)
class ShrinkageLDAEvaluation:
    probabilities: np.ndarray
    predictions: np.ndarray
    outer_thresholds: np.ndarray
    metrics: dict[str, float]
    outer_representations: tuple[str, ...]
    n_participants: int
    selection_protocol: str


def _matrix(value: object, *, name: str, min_rows: int = 2) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < 1
            or array.dtype.kind not in {"i", "u", "f"}):
        raise ValueError(f"{name} must be a nonempty participant-by-feature matrix")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def fit_fold_preprocessor(
    train: np.ndarray,
    test: np.ndarray,
    *,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> FoldFeatures:
    """Fit standardization and ordered correlation filtering on train only."""
    train = _matrix(train, name="train")
    test = _matrix(test, name="test", min_rows=1)
    if test.shape[1] != train.shape[1]:
        raise ValueError("train and test feature counts differ")
    if (not isinstance(correlation_threshold, (int, float, np.integer, np.floating))
            or isinstance(correlation_threshold, (bool, np.bool_))
            or not 0 < float(correlation_threshold) < 1):
        raise ValueError("correlation threshold must be between zero and one")
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    standardized_train = (train - mean) / scale
    standardized_test = (test - mean) / scale

    feature_count = standardized_train.shape[1]
    standard_deviations = standardized_train.std(axis=0, ddof=0)
    active = np.flatnonzero(standard_deviations > 0)
    correlations = np.zeros((feature_count, feature_count), dtype=np.float64)
    if active.size == 1:
        correlations[active[0], active[0]] = 1.0
    elif active.size > 1:
        active_correlations = np.corrcoef(
            standardized_train[:, active], rowvar=False
        )
        correlations[np.ix_(active, active)] = active_correlations

    kept: list[int] = []
    for candidate in range(standardized_train.shape[1]):
        drop = False
        if standard_deviations[candidate] > 0:
            for prior in kept:
                if standard_deviations[prior] == 0:
                    continue
                correlation = float(correlations[candidate, prior])
                if np.isfinite(correlation) and abs(correlation) > float(correlation_threshold):
                    drop = True
                    break
        if not drop:
            kept.append(candidate)
    if not kept:
        raise ValueError("correlation filtering removed every feature")
    indices = np.asarray(kept, dtype=np.int64)
    return FoldFeatures(
        train=_frozen(standardized_train[:, indices]),
        test=_frozen(standardized_test[:, indices]),
        kept_indices=_frozen(indices),
        train_mean=_frozen(mean),
        train_scale=_frozen(scale),
    )


def _validate_dataset(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
    if not isinstance(representations, Mapping) or not representations:
        raise ValueError("at least one named representation is required")
    matrices = {}
    row_count = None
    for name, values in representations.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("representation names must be canonical strings")
        matrix = _matrix(values, name=name)
        if row_count is None:
            row_count = matrix.shape[0]
        if matrix.shape[0] != row_count:
            raise ValueError("representations have different participant counts")
        matrices[name] = matrix
    labels = np.asarray(labels)
    if (labels.shape != (row_count,) or labels.dtype.kind not in {"i", "u"}
            or not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2):
        raise ValueError("labels must contain both binary classes once per participant")
    labels = labels.astype(np.int64, copy=True)
    groups = tuple(group_ids)
    if (len(groups) != row_count or len(set(groups)) != row_count
            or any(not isinstance(value, str) or not value.startswith("grp_")
                   for value in groups)):
        raise ValueError("group_ids must be unique opaque participant identifiers")
    return matrices, labels, groups


def _fit_predict(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    candidate: Candidate,
) -> np.ndarray:
    fold = fit_fold_preprocessor(
        matrix[train_indices], matrix[test_indices],
        correlation_threshold=CORRELATION_THRESHOLD,
    )
    train_labels = labels[train_indices]
    return _fit_preprocessed(fold, train_labels, candidate)


def _fit_preprocessed(
    fold: FoldFeatures,
    train_labels: np.ndarray,
    candidate: Candidate,
    *,
    class_weight: str | None = None,
) -> np.ndarray:
    """Fit one candidate on a candidate-independent preprocessed fold."""
    if not isinstance(fold, FoldFeatures) or not isinstance(candidate, Candidate):
        raise ValueError("preprocessed fitting requires a fold and frozen candidate")
    train_labels = np.asarray(train_labels)
    if (train_labels.shape != (fold.train.shape[0],)
            or train_labels.dtype.kind not in {"i", "u"}
            or not np.isin(train_labels, (0, 1)).all()):
        raise ValueError("preprocessed training labels are invalid")
    if len(np.unique(train_labels)) != 2:
        raise ValueError("every training fold must contain both classes")
    if class_weight not in {None, "balanced"}:
        raise ValueError("class_weight must be absent or exactly balanced")
    classifier = LogisticRegression(
        C=candidate.c,
        penalty=candidate.penalty,
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    classifier.fit(fold.train, train_labels)
    probabilities = classifier.predict_proba(fold.test)[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("classifier emitted invalid probabilities")
    return probabilities.astype(np.float64, copy=False)


def _fit_predict_shrinkage_lda(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    fold = fit_fold_preprocessor(
        matrix[train_indices], matrix[test_indices],
        correlation_threshold=CORRELATION_THRESHOLD,
    )
    train_labels = labels[train_indices]
    if len(np.unique(train_labels)) != 2:
        raise ValueError("every shrinkage-LDA training fold must contain both classes")
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(fold.train, train_labels)
    probabilities = classifier.predict_proba(fold.test)[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("shrinkage LDA emitted invalid probabilities")
    return probabilities.astype(np.float64, copy=False)


def recompute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (labels.ndim != 1 or probabilities.shape != labels.shape
            or labels.dtype.kind not in {"i", "u"}
            or not np.isin(labels, (0, 1)).all()
            or len(np.unique(labels)) != 2
            or probabilities.dtype.kind != "f"
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("metrics require finite binary labels and probabilities")
    labels = labels.astype(np.int64, copy=False)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "accuracy": float((tp + tn) / len(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _metrics_from_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    if (labels.ndim != 1 or probabilities.shape != labels.shape
            or predictions.shape != labels.shape
            or labels.dtype.kind not in {"i", "u"}
            or predictions.dtype.kind not in {"i", "u"}
            or not np.isin(labels, (0, 1)).all()
            or not np.isin(predictions, (0, 1)).all()
            or len(np.unique(labels)) != 2
            or probabilities.dtype.kind != "f"
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("threshold metrics require binary labels and predictions")
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "accuracy": float((tp + tn) / len(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def select_balanced_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Choose a deterministic balanced-accuracy threshold from inner OOF rows."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    recompute_binary_metrics(labels, probabilities)
    candidates = np.unique(np.concatenate((
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        probabilities.astype(np.float64, copy=False),
    )))
    best_threshold = None
    best_key = None
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = _metrics_from_predictions(labels, probabilities, predictions)
        key = (
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    if best_threshold is None:
        raise AssertionError("threshold selection produced no candidate")
    return best_threshold


def participant_stratified_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    predictions: np.ndarray | None = None,
    replicates: int = 5000,
    seed: int = 20260814,
) -> dict[str, object]:
    """Return deterministic class-stratified participant bootstrap intervals."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    recompute_binary_metrics(labels, probabilities)
    if predictions is None:
        predictions = (probabilities >= 0.5).astype(np.int64)
    else:
        predictions = np.asarray(predictions)
    _metrics_from_predictions(labels, probabilities, predictions)
    if (isinstance(replicates, (bool, np.bool_)) or not isinstance(
            replicates, (int, np.integer)) or not 1 <= int(replicates) <= 100_000):
        raise ValueError("bootstrap replicates must be an integer from 1 to 100000")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("bootstrap seed must be a nonnegative integer")
    replicates = int(replicates)
    seed = int(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    metric_names = (
        "accuracy", "auroc", "balanced_accuracy", "sensitivity", "specificity"
    )
    samples = {name: np.empty(replicates, dtype=np.float64) for name in metric_names}
    generator = np.random.default_rng(seed)
    for index in range(replicates):
        selected = np.concatenate((
            generator.choice(negative, size=len(negative), replace=True),
            generator.choice(positive, size=len(positive), replace=True),
        ))
        metrics = _metrics_from_predictions(
            labels[selected], probabilities[selected], predictions[selected]
        )
        for name in metric_names:
            samples[name][index] = metrics[name]
    return {
        "method": "participant_stratified_percentile_bootstrap",
        "replicates": replicates,
        "valid_replicates": replicates,
        "seed": seed,
        "interval_95": {
            name: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for name, values in samples.items()
        },
    }


def evaluate_fixed_loso(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
    candidate: Candidate,
) -> FixedEvaluation:
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    if not isinstance(candidate, Candidate) or candidate.representation not in matrices:
        raise ValueError("candidate representation is unavailable")
    count = len(groups)
    probabilities = np.empty(count, dtype=np.float64)
    all_indices = np.arange(count, dtype=np.int64)
    for held in range(count):
        train = all_indices[all_indices != held]
        probabilities[held] = _fit_predict(
            matrices[candidate.representation], labels, train,
            np.asarray([held], dtype=np.int64), candidate,
        )[0]
    return FixedEvaluation(
        candidate=candidate,
        probabilities=_frozen(probabilities),
        metrics=recompute_binary_metrics(labels, probabilities),
        n_participants=count,
        selection_protocol="fixed_participant_loso",
    )


def _candidate_order(representations: Mapping[str, np.ndarray]):
    for representation in sorted(representations):
        for penalty in PENALTIES:
            for c_value in C_GRID:
                yield Candidate(representation, penalty, c_value)


def _selection_key(result: FixedEvaluation):
    metrics = result.metrics
    penalty_preference = 1 if result.candidate.penalty == "l2" else 0
    return (
        metrics["accuracy"], metrics["auroc"], metrics["balanced_accuracy"],
        -C_GRID.index(result.candidate.c), penalty_preference,
        result.candidate.representation,
    )


def select_paper_like_candidate(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> FixedEvaluation:
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    count = len(groups)
    all_indices = np.arange(count, dtype=np.int64)
    fold_cache: dict[str, tuple[tuple[np.ndarray, FoldFeatures], ...]] = {}
    for name, matrix in matrices.items():
        cached = []
        for held in range(count):
            train = all_indices[all_indices != held]
            fold = fit_fold_preprocessor(
                matrix[train], matrix[[held]],
                correlation_threshold=CORRELATION_THRESHOLD,
            )
            cached.append((train, fold))
        fold_cache[name] = tuple(cached)

    results = []
    for candidate in _candidate_order(matrices):
        probabilities = np.empty(count, dtype=np.float64)
        for held, (train, fold) in enumerate(fold_cache[candidate.representation]):
            probabilities[held] = _fit_preprocessed(
                fold, labels[train], candidate
            )[0]
        results.append(FixedEvaluation(
            candidate=candidate,
            probabilities=_frozen(probabilities),
            metrics=recompute_binary_metrics(labels, probabilities),
            n_participants=count,
            selection_protocol="fixed_participant_loso",
        ))
    best = max(results, key=_selection_key)
    return replace(best, selection_protocol="same_oof_candidate_search_descriptive")


def select_oof_candidate_with_threshold(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
    *,
    class_weight: str | None = None,
) -> ThresholdCandidateSelection:
    """Jointly select a candidate and threshold from participant OOF predictions."""
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    count = len(groups)
    all_indices = np.arange(count, dtype=np.int64)
    fold_cache: dict[str, tuple[tuple[np.ndarray, FoldFeatures], ...]] = {}
    for name, matrix in matrices.items():
        cached = []
        for held in range(count):
            train = all_indices[all_indices != held]
            cached.append((train, fit_fold_preprocessor(
                matrix[train], matrix[[held]],
                correlation_threshold=CORRELATION_THRESHOLD,
            )))
        fold_cache[name] = tuple(cached)
    results = []
    for candidate in _candidate_order(matrices):
        probabilities = np.empty(count, dtype=np.float64)
        for held, (train, fold) in enumerate(fold_cache[candidate.representation]):
            probabilities[held] = _fit_preprocessed(
                fold, labels[train], candidate, class_weight=class_weight
            )[0]
        threshold = select_balanced_threshold(labels, probabilities)
        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = _metrics_from_predictions(labels, probabilities, predictions)
        results.append(ThresholdCandidateSelection(
            candidate=candidate,
            probabilities=_frozen(probabilities),
            predictions=_frozen(predictions),
            threshold=threshold,
            metrics=metrics,
            n_participants=count,
            selection_protocol=(
                "same_oof_balanced_candidate_and_threshold_search_descriptive"
                if class_weight == "balanced" else
                "same_oof_candidate_and_threshold_search_descriptive"
            ),
        ))
    def key(result: ThresholdCandidateSelection):
        return (
            result.metrics["balanced_accuracy"],
            result.metrics["accuracy"],
            result.metrics["auroc"],
            -abs(result.threshold - 0.5),
            -C_GRID.index(result.candidate.c),
            1 if result.candidate.penalty == "l2" else 0,
            result.candidate.representation,
        )
    return max(results, key=key)


def evaluate_nested_loso(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> NestedEvaluation:
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    count = len(groups)
    all_indices = np.arange(count, dtype=np.int64)
    probabilities = np.empty(count, dtype=np.float64)
    outer_candidates = []
    for held in range(count):
        outer_train = all_indices[all_indices != held]
        inner_representations = {
            name: matrix[outer_train] for name, matrix in matrices.items()
        }
        inner_labels = labels[outer_train]
        inner_groups = tuple(groups[index] for index in outer_train)
        selected = select_paper_like_candidate(
            inner_representations, inner_labels, inner_groups
        ).candidate
        outer_candidates.append(selected)
        probabilities[held] = _fit_predict(
            matrices[selected.representation], labels, outer_train,
            np.asarray([held], dtype=np.int64), selected,
        )[0]
    return NestedEvaluation(
        probabilities=_frozen(probabilities),
        metrics=recompute_binary_metrics(labels, probabilities),
        outer_candidates=tuple(outer_candidates),
        n_participants=count,
        selection_protocol="nested_participant_loso",
    )


def evaluate_nested_loso_with_threshold(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> ThresholdedNestedEvaluation:
    """Select representation, regularization, and threshold inside each outer fold."""
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    count = len(groups)
    all_indices = np.arange(count, dtype=np.int64)
    probabilities = np.empty(count, dtype=np.float64)
    predictions = np.empty(count, dtype=np.int64)
    thresholds = np.empty(count, dtype=np.float64)
    outer_candidates = []
    for held in range(count):
        outer_train = all_indices[all_indices != held]
        inner_representations = {
            name: matrix[outer_train] for name, matrix in matrices.items()
        }
        inner_labels = labels[outer_train]
        inner_groups = tuple(groups[index] for index in outer_train)
        inner_result = select_oof_candidate_with_threshold(
            inner_representations, inner_labels, inner_groups
        )
        selected = inner_result.candidate
        threshold = inner_result.threshold
        outer_candidates.append(selected)
        thresholds[held] = threshold
        probabilities[held] = _fit_predict(
            matrices[selected.representation], labels, outer_train,
            np.asarray([held], dtype=np.int64), selected,
        )[0]
        predictions[held] = int(probabilities[held] >= threshold)
    return ThresholdedNestedEvaluation(
        probabilities=_frozen(probabilities),
        predictions=_frozen(predictions),
        outer_thresholds=_frozen(thresholds),
        metrics=_metrics_from_predictions(labels, probabilities, predictions),
        outer_candidates=tuple(outer_candidates),
        n_participants=count,
        selection_protocol="nested_participant_loso_with_inner_oof_threshold",
    )


def evaluate_nested_balanced_logistic(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> ThresholdedNestedEvaluation:
    """Run class-balanced Logistic with candidate and threshold nested by person."""
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    count = len(groups)
    all_indices = np.arange(count, dtype=np.int64)
    probabilities = np.empty(count, dtype=np.float64)
    predictions = np.empty(count, dtype=np.int64)
    thresholds = np.empty(count, dtype=np.float64)
    outer_candidates = []
    for held in range(count):
        outer_train = all_indices[all_indices != held]
        inner_representations = {
            name: matrix[outer_train] for name, matrix in matrices.items()
        }
        inner_labels = labels[outer_train]
        inner_groups = tuple(groups[index] for index in outer_train)
        inner_result = select_oof_candidate_with_threshold(
            inner_representations,
            inner_labels,
            inner_groups,
            class_weight="balanced",
        )
        selected = inner_result.candidate
        threshold = inner_result.threshold
        outer_candidates.append(selected)
        thresholds[held] = threshold
        fold = fit_fold_preprocessor(
            matrices[selected.representation][outer_train],
            matrices[selected.representation][[held]],
            correlation_threshold=CORRELATION_THRESHOLD,
        )
        probabilities[held] = _fit_preprocessed(
            fold,
            labels[outer_train],
            selected,
            class_weight="balanced",
        )[0]
        predictions[held] = int(probabilities[held] >= threshold)
    return ThresholdedNestedEvaluation(
        probabilities=_frozen(probabilities),
        predictions=_frozen(predictions),
        outer_thresholds=_frozen(thresholds),
        metrics=_metrics_from_predictions(labels, probabilities, predictions),
        outer_candidates=tuple(outer_candidates),
        n_participants=count,
        selection_protocol=(
            "nested_participant_loso_balanced_logistic_with_inner_oof_threshold"
        ),
    )


def evaluate_nested_shrinkage_lda(
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> ShrinkageLDAEvaluation:
    """Select one representation and threshold inside outer LOSO for shrinkage LDA."""
    matrices, labels, groups = _validate_dataset(representations, labels, group_ids)
    count = len(groups)
    all_indices = np.arange(count, dtype=np.int64)
    probabilities = np.empty(count, dtype=np.float64)
    predictions = np.empty(count, dtype=np.int64)
    thresholds = np.empty(count, dtype=np.float64)
    outer_representations = []
    representation_order = tuple(sorted(matrices))
    for held in range(count):
        outer_train = all_indices[all_indices != held]
        inner_labels = labels[outer_train]
        inner_groups = tuple(groups[index] for index in outer_train)
        # Validation preserves uniqueness and two classes for every inner endpoint.
        _validate_dataset(
            {name: matrix[outer_train] for name, matrix in matrices.items()},
            inner_labels,
            inner_groups,
        )
        inner_indices = np.arange(len(outer_train), dtype=np.int64)
        inner_results = {}
        for name in representation_order:
            inner_matrix = matrices[name][outer_train]
            inner_probabilities = np.empty(len(outer_train), dtype=np.float64)
            for inner_held in range(len(outer_train)):
                inner_train = inner_indices[inner_indices != inner_held]
                inner_probabilities[inner_held] = _fit_predict_shrinkage_lda(
                    inner_matrix,
                    inner_labels,
                    inner_train,
                    np.asarray([inner_held], dtype=np.int64),
                )[0]
            threshold = select_balanced_threshold(inner_labels, inner_probabilities)
            inner_predictions = (inner_probabilities >= threshold).astype(np.int64)
            inner_results[name] = (
                inner_probabilities,
                _metrics_from_predictions(
                    inner_labels, inner_probabilities, inner_predictions
                ),
                threshold,
            )
        selected_name = max(
            representation_order,
            key=lambda name: (
                inner_results[name][1]["accuracy"],
                inner_results[name][1]["auroc"],
                inner_results[name][1]["balanced_accuracy"],
                -representation_order.index(name),
            ),
        )
        inner_probabilities = inner_results[selected_name][0]
        threshold = float(inner_results[selected_name][2])
        outer_representations.append(selected_name)
        thresholds[held] = threshold
        probabilities[held] = _fit_predict_shrinkage_lda(
            matrices[selected_name], labels, outer_train,
            np.asarray([held], dtype=np.int64),
        )[0]
        predictions[held] = int(probabilities[held] >= threshold)
    return ShrinkageLDAEvaluation(
        probabilities=_frozen(probabilities),
        predictions=_frozen(predictions),
        outer_thresholds=_frozen(thresholds),
        metrics=_metrics_from_predictions(labels, probabilities, predictions),
        outer_representations=tuple(outer_representations),
        n_participants=count,
        selection_protocol=(
            "nested_participant_loso_shrinkage_lda_with_inner_oof_threshold"
        ),
    )


def build_public_report(
    *,
    endpoint: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    protocol: str,
    representation: str,
) -> dict[str, object]:
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in (endpoint, protocol, representation)):
        raise ValueError("public report strings must be canonical")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    metrics = recompute_binary_metrics(labels, probabilities)
    return {
        "schema_version": "neuroface_als_benchmark_public_v1",
        "endpoint": endpoint,
        "protocol": protocol,
        "representation": representation,
        "counts": {
            "participants": int(len(labels)),
            "positive": int(np.sum(labels == 1)),
            "negative": int(np.sum(labels == 0)),
        },
        "threshold": 0.5,
        "metrics": metrics,
        "published_comparator": {
            "endpoint": "als_vs_healthy_spread_minimum_pyfeat_au",
            "accuracy": PAPER_ACCURACY,
            "auroc": PAPER_AUROC,
            "comparison_status": "descriptive_not_external_superiority",
        },
    }


__all__ = [
    "C_GRID", "CORRELATION_THRESHOLD", "Candidate", "FixedEvaluation",
    "FoldFeatures", "NestedEvaluation", "ShrinkageLDAEvaluation",
    "ThresholdCandidateSelection", "ThresholdedNestedEvaluation",
    "PAPER_ACCURACY", "PAPER_AUROC",
    "build_public_report", "evaluate_fixed_loso", "evaluate_nested_loso",
    "evaluate_nested_balanced_logistic", "evaluate_nested_loso_with_threshold",
    "evaluate_nested_shrinkage_lda",
    "fit_fold_preprocessor",
    "participant_stratified_bootstrap", "recompute_binary_metrics",
    "select_balanced_threshold",
    "select_oof_candidate_with_threshold", "select_paper_like_candidate",
]
