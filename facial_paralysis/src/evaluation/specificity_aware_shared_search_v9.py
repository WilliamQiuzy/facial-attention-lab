"""Participant-disjoint specificity-aware evaluation for shared V9."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2,
    _model_inputs,
    _scaled,
    _tensor,
)
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    fit_clinical_scaler,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.specificity_aware_candidate_registry_v9 import (
    SpecificityCandidateV9,
    candidate_registry_v9,
)
from src.models.specificity_aware_shared_router_v9 import (
    SpecificityAwareSharedRouterV9,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_CALIBRATION_SENSITIVITY = 0.90
_RANKING_SENSITIVITY_FLOOR = 0.85
_COMPARATOR_NONINFERIORITY = 0.01


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class SpecificityEvaluationV9:
    probabilities: np.ndarray
    calibrated_predictions: np.ndarray
    fixed_metrics: dict[str, dict[str, float]]
    calibrated_metrics: dict[str, dict[str, float]]
    thresholds_by_source: dict[str, tuple[float, ...]]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]


def _validate_binary_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or len(np.unique(labels)) != 2
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("binary operating points require finite two-class evidence")
    return labels.astype(np.int64, copy=False), probabilities


def select_training_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_sensitivity: float,
) -> float:
    """Select an operating point from training predictions only."""
    labels, probabilities = _validate_binary_arrays(labels, probabilities)
    if (
        isinstance(min_sensitivity, bool)
        or not isinstance(min_sensitivity, (int, float))
        or not np.isfinite(float(min_sensitivity))
        or float(min_sensitivity) <= 0.0
        or float(min_sensitivity) > 1.0
    ):
        raise ValueError("minimum sensitivity must be in (0, 1]")
    positive = labels == 1
    negative = labels == 0
    candidates = np.unique(np.concatenate((probabilities, np.asarray([0.5]))))
    feasible = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        sensitivity = float(np.mean(predicted[positive]))
        if sensitivity + 1e-12 < float(min_sensitivity):
            continue
        specificity = float(np.mean(~predicted[negative]))
        accuracy = float(np.mean(predicted == positive))
        feasible.append((specificity, accuracy, float(threshold)))
    if not feasible:
        raise RuntimeError("no training-only threshold satisfies the sensitivity floor")
    return max(feasible)[2]


def calibrated_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels, probabilities = _validate_binary_arrays(labels, probabilities)
    predictions = np.asarray(predictions)
    if predictions.shape != labels.shape or predictions.dtype != np.dtype(bool):
        raise ValueError("calibrated predictions must be an aligned boolean vector")
    positive = labels == 1
    negative = labels == 0
    sensitivity = float(np.mean(predictions[positive]))
    specificity = float(np.mean(~predictions[negative]))
    return {
        "accuracy": float(np.mean(predictions == positive)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": float(np.mean(np.square(probabilities - labels))),
    }


def _training_weights(
    labels: np.ndarray,
    sources: tuple[str, ...],
    control_cost: float,
) -> np.ndarray:
    weights = np.array(source_class_balanced_weights(labels, sources), copy=True)
    weights[labels == 0] *= control_cost
    weights /= weights.sum()
    if not np.isclose(weights.sum(), 1.0) or np.any(weights <= 0.0):
        raise RuntimeError("V9 participant weights failed normalization")
    return weights


def _probabilities(
    model: SpecificityAwareSharedRouterV9,
    inputs: tuple[torch.Tensor, ...],
    task_codes: torch.Tensor,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tokens = model.shared_action_tokens(*inputs)
        logits = model.routed_logits(tokens, inputs[-2], task_codes)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


def evaluate_specificity_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: SpecificityCandidateV9,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> SpecificityEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not SpecificityCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid V9 evaluation configuration")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested V9 evaluation device is unavailable")

    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    calibrated_predictions = np.zeros(len(base.labels), dtype=bool)
    thresholds: dict[str, list[float]] = {source: [] for source in SOURCES}
    covered: set[str] = set()

    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = SpecificityAwareSharedRouterV9(candidate).to(runtime)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-3
        )
        train_sources = tuple(base.sources[index] for index in train)
        train_labels_array = base.labels[train]
        weights = _tensor(
            _training_weights(
                train_labels_array, train_sources, candidate.control_cost
            ).astype(np.float32),
            runtime,
        )
        labels = _tensor(train_labels_array.astype(np.float32), runtime)
        tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        train_inputs = _model_inputs(dataset, original, mirrored, train, runtime)

        if fold_index == 0:
            for source in SOURCES:
                local = torch.tensor([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=torch.long, device=runtime)
                model.zero_grad(set_to_none=True)
                local_inputs = tuple(value.index_select(0, local) for value in train_inputs)
                tokens = model.shared_action_tokens(*local_inputs)
                local_logits = model.routed_logits(
                    tokens, local_inputs[-2], tasks.index_select(0, local)
                )
                loss = F.binary_cross_entropy_with_logits(
                    local_logits, labels.index_select(0, local)
                )
                loss.backward()
                clinical = model.base.base.backbone.clinical_encoder[0].weight.grad
                patient = model.base.base.backbone.patient_projection.weight.grad
                if (
                    clinical is None
                    or patient is None
                    or float(clinical.norm()) <= 0.0
                    or float(patient.norm()) <= 0.0
                ):
                    raise RuntimeError("a source failed the shared V9 gradient audit")
                covered.add(source)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tokens = model.shared_action_tokens(*train_inputs)
            endpoint, universal = model.patient_embeddings(
                tokens, train_inputs[-2], tasks
            )
            task_logits = model.base.base.task_logits_from_embedding(endpoint, tasks)
            universal_logits = model.base.base.universal_head(universal).squeeze(-1)
            shared_logits = (
                (1.0 - candidate.universal_blend) * task_logits
                + candidate.universal_blend * universal_logits
            )
            healthy_blend = {
                "off": 0.0, "compact": 0.15, "compact_margin": 0.25,
            }[candidate.healthy_mode]
            routed = shared_logits if healthy_blend == 0.0 else (
                (1.0 - healthy_blend) * shared_logits
                + healthy_blend * model.normality_logits(universal)
            )
            losses = F.binary_cross_entropy_with_logits(
                routed, labels, reduction="none"
            ) + 0.5 * F.binary_cross_entropy_with_logits(
                universal_logits, labels, reduction="none"
            )
            loss = torch.sum(losses * weights)
            loss = loss + model.normal_reference_loss(universal, labels, weights)
            loss = loss + model.control_alignment_loss(
                universal, labels, tasks, weights
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V9 training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_probabilities = _probabilities(model, train_inputs, tasks)
        held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
        held_tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[base.sources[index]] for index in held
        ], dtype=np.int64), runtime)
        held_probabilities = _probabilities(model, held_inputs, held_tasks)
        probabilities[held] = held_probabilities
        for source in SOURCES:
            local_train = np.asarray([
                observed == source for observed in train_sources
            ])
            threshold = select_training_threshold(
                train_labels_array[local_train],
                train_probabilities[local_train],
                _CALIBRATION_SENSITIVITY,
            )
            thresholds[source].append(threshold)
            local_held_positions = np.asarray([
                base.sources[index] == source for index in held
            ])
            calibrated_predictions[held[local_held_positions]] = (
                held_probabilities[local_held_positions] >= threshold
            )
        del model, optimizer

    if (
        not np.isfinite(probabilities).all()
        or covered != set(SOURCES)
        or any(len(values) != len(folds) for values in thresholds.values())
    ):
        raise RuntimeError("V9 evaluation did not cover the frozen participant set")
    fixed_metrics = {}
    calibrated_metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in base.sources])
        fixed_metrics[source] = binary_metrics(
            base.labels[selected], probabilities[selected]
        )
        calibrated_metrics[source] = calibrated_binary_metrics(
            base.labels[selected], probabilities[selected],
            calibrated_predictions[selected],
        )
    return SpecificityEvaluationV9(
        probabilities=_immutable(probabilities),
        calibrated_predictions=_immutable(calibrated_predictions),
        fixed_metrics=fixed_metrics,
        calibrated_metrics=calibrated_metrics,
        thresholds_by_source={
            source: tuple(float(value) for value in thresholds[source])
            for source in SOURCES
        },
        model_fits=len(folds),
        shared_gradient_sources=SOURCES,
    )


def _is_feasible(
    result: SpecificityEvaluationV9,
    comparator: SpecificityEvaluationV9,
) -> bool:
    for source in SOURCES:
        observed = result.calibrated_metrics[source]
        baseline = comparator.calibrated_metrics[source]
        if (
            observed["sensitivity"] + 1e-12 < _RANKING_SENSITIVITY_FLOOR
            or observed["accuracy"] + _COMPARATOR_NONINFERIORITY + 1e-12
            < baseline["accuracy"]
            or observed["auroc"] + _COMPARATOR_NONINFERIORITY + 1e-12
            < baseline["auroc"]
        ):
            return False
    return True


def rank_specificity_results(
    results: dict[str, SpecificityEvaluationV9],
    *,
    comparator_id: str = "SSR9-000",
) -> tuple[str, ...]:
    expected = {row.candidate_id for row in candidate_registry_v9()}
    if (
        type(results) is not dict
        or set(results) != expected
        or comparator_id not in expected
        or any(type(value) is not SpecificityEvaluationV9 for value in results.values())
    ):
        raise ValueError("V9 ranking requires every frozen candidate and comparator")
    comparator = results[comparator_id]

    def key(candidate_id: str):
        result = results[candidate_id]
        metrics = result.calibrated_metrics
        specificity = [metrics[source]["specificity"] for source in SOURCES]
        auroc = [metrics[source]["auroc"] for source in SOURCES]
        accuracy = [metrics[source]["accuracy"] for source in SOURCES]
        balanced = [metrics[source]["balanced_accuracy"] for source in SOURCES]
        return (
            not _is_feasible(result, comparator),
            -min(specificity),
            -min(auroc),
            -min(accuracy),
            -min(balanced),
            -float(np.mean(accuracy)),
            candidate_id,
        )

    return tuple(sorted(expected, key=key))


__all__ = [
    "SpecificityEvaluationV9",
    "calibrated_binary_metrics",
    "evaluate_specificity_candidate",
    "rank_specificity_results",
    "select_training_threshold",
]
