"""Frozen participant-disjoint and leave-one-source-out shared V9 evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.distilled_shared_search_v9 import (
    configure_deterministic_training_v9,
)
from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2,
    _model_inputs,
    _scaled,
    _tensor,
)
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    fit_clinical_scaler,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.anatomical_relational_router_v9 import (
    AnatomicalRelationalRouterV9,
    candidate_registry_v9 as anatomical_candidate_registry,
)
from src.models.literature_grounded_candidate_registry_v9 import (
    LiteratureGroundedCandidateV9,
    candidate_registry_v9,
)
from src.models.residual_shared_router_v8 import candidate_registry_v8
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)
from src.training.clinical_kinematic_auxiliary_v9 import (
    ClinicalKinematicAuxiliaryHeadV9,
    clinical_kinematic_auxiliary_loss,
    clinical_kinematic_targets,
    fit_kinematic_target_scaler,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_FROZEN_FEATURE_NAMES = candidate_feature_names(LANDMARK_MI_110D)


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class LiteratureGroundedEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    loso_metrics: dict[str, dict[str, float]]
    loso_train_sources: tuple[tuple[str, tuple[str, ...]], ...]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    task_specific_parameter_fraction: float


def _anatomical_candidate(relation_enabled: bool):
    matched = tuple(
        row for row in anatomical_candidate_registry()
        if row.relation_enabled is relation_enabled
    )
    if len(matched) != 1:
        raise RuntimeError("the anatomical candidate mapping is ambiguous")
    return matched[0]


def _source_indices(sources: tuple[str, ...], runtime: torch.device):
    return {
        source: torch.tensor([
            index for index, observed in enumerate(sources) if observed == source
        ], dtype=torch.long, device=runtime)
        for source in SOURCES
        if source in sources
    }


def _fit_model(
    dataset: MedicalSharedDatasetV2,
    candidate: LiteratureGroundedCandidateV9,
    train: np.ndarray,
    *,
    epochs: int,
    local_seed: int,
    runtime: torch.device,
    audit_sources: bool,
):
    base = dataset.base
    torch.manual_seed(local_seed)
    if runtime.type == "cuda":
        torch.cuda.manual_seed_all(local_seed)
    scaler = fit_clinical_scaler(base, train)
    original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
    mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
    model = AnatomicalRelationalRouterV9(
        _anatomical_candidate(candidate.relation_enabled)
    ).to(runtime)
    auxiliary = (
        ClinicalKinematicAuxiliaryHeadV9().to(runtime)
        if candidate.auxiliary_weight > 0.0 else None
    )
    parameters = tuple(model.parameters()) + (
        tuple(auxiliary.parameters()) if auxiliary is not None else ()
    )
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-3)
    train_sources = tuple(base.sources[index] for index in train)
    weights = _tensor(
        source_class_balanced_weights(
            base.labels[train], train_sources
        ).astype(np.float32), runtime,
    )
    labels = _tensor(base.labels[train].astype(np.float32), runtime)
    tasks = _tensor(np.asarray([
        _SOURCE_TASK_CODE[source] for source in train_sources
    ], dtype=np.int64), runtime)
    inputs = _model_inputs(dataset, original, mirrored, train, runtime)
    target_values = None
    target_scaler = None
    if auxiliary is not None:
        target_values = clinical_kinematic_targets(
            _tensor(base.clinical_original[train], runtime),
            _tensor(base.clinical_mirrored[train], runtime),
            inputs[-2],
            _FROZEN_FEATURE_NAMES,
        )
        target_scaler = fit_kinematic_target_scaler(target_values, inputs[-2])

    covered: set[str] = set()
    if audit_sources:
        for source, local in _source_indices(train_sources, runtime).items():
            model.zero_grad(set_to_none=True)
            local_inputs = tuple(value.index_select(0, local) for value in inputs)
            local_tasks = tasks.index_select(0, local)
            local_labels = labels.index_select(0, local)
            tokens = model.shared_action_tokens(*local_inputs)
            routed, _universal = model.routed_and_universal_logits(
                tokens, local_inputs[-2], local_tasks
            )
            F.binary_cross_entropy_with_logits(routed, local_labels).backward()
            clinical_gradient = model.base.base.backbone.clinical_encoder[0].weight.grad
            patient_gradient = model.base.base.backbone.patient_projection.weight.grad
            if (
                clinical_gradient is None or patient_gradient is None
                or float(clinical_gradient.norm()) <= 0.0
                or float(patient_gradient.norm()) <= 0.0
            ):
                raise RuntimeError("a source failed the genuinely shared gradient audit")
            covered.add(source)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tokens = model.shared_action_tokens(*inputs)
        routed, universal = model.routed_and_universal_logits(
            tokens, inputs[-2], tasks
        )
        losses = F.binary_cross_entropy_with_logits(
            routed, labels, reduction="none"
        ) + 0.5 * F.binary_cross_entropy_with_logits(
            universal, labels, reduction="none"
        )
        objective = torch.sum(losses * weights)
        if auxiliary is not None:
            objective = objective + candidate.auxiliary_weight * (
                clinical_kinematic_auxiliary_loss(
                    auxiliary, tokens, target_values, inputs[-2], target_scaler
                )
            )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return model, original, mirrored, covered


def _predict_routed(
    model: AnatomicalRelationalRouterV9,
    dataset: MedicalSharedDatasetV2,
    original: np.ndarray,
    mirrored: np.ndarray,
    held: np.ndarray,
    runtime: torch.device,
) -> np.ndarray:
    held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
    held_tasks = _tensor(np.asarray([
        _SOURCE_TASK_CODE[dataset.base.sources[index]] for index in held
    ], dtype=np.int64), runtime)
    model.eval()
    with torch.no_grad():
        tokens = model.shared_action_tokens(*held_inputs)
        logits = model.routed_logits(tokens, held_inputs[-2], held_tasks)
    return torch.sigmoid(logits).cpu().numpy()


def _loso_metrics(
    dataset: MedicalSharedDatasetV2,
    candidate: LiteratureGroundedCandidateV9,
    *,
    epochs: int,
    seed: int,
    runtime: torch.device,
):
    base = dataset.base
    results = {}
    training_sources = []
    for target_index, target_source in enumerate(SOURCES):
        train = np.asarray([
            index for index, source in enumerate(base.sources)
            if source != target_source
        ], dtype=np.int64)
        held = np.asarray([
            index for index, source in enumerate(base.sources)
            if source == target_source
        ], dtype=np.int64)
        model, original, mirrored, _ = _fit_model(
            dataset, candidate, train, epochs=epochs,
            local_seed=seed * 1009 + 100 + target_index,
            runtime=runtime, audit_sources=False,
        )
        held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
        model.eval()
        with torch.no_grad():
            tokens = model.shared_action_tokens(*held_inputs)
            probabilities = torch.sigmoid(
                model.universal_logits(tokens, held_inputs[-2])
            ).cpu().numpy()
        results[target_source] = binary_metrics(base.labels[held], probabilities)
        observed_training = tuple(
            source for source in SOURCES if source != target_source
        )
        training_sources.append((target_source, observed_training))
        del model
    return results, tuple(training_sources)


def evaluate_literature_grounded_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: LiteratureGroundedCandidateV9,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> LiteratureGroundedEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not LiteratureGroundedCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid literature-grounded V9 evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("literature-grounded V9 runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    base = dataset.base
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    if candidate.candidate_id == "LGS9-000":
        residual = evaluate_residual_candidate(
            dataset,
            next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"),
            epochs=epochs, n_splits=n_splits, seed=seed, device=device,
        )
        probabilities[:] = residual.probabilities
        metrics = residual.metrics
        covered.update(residual.shared_gradient_sources)
        fraction = AnatomicalRelationalRouterV9(
            _anatomical_candidate(False)
        ).task_specific_parameter_fraction()
    else:
        fraction = np.nan
        for fold_index, (train, held) in enumerate(folds):
            model, original, mirrored, audited = _fit_model(
                dataset, candidate, train, epochs=epochs,
                local_seed=seed * 1009 + fold_index,
                runtime=runtime, audit_sources=fold_index == 0,
            )
            probabilities[held] = _predict_routed(
                model, dataset, original, mirrored, held, runtime
            )
            covered.update(audited)
            fraction = model.task_specific_parameter_fraction()
            del model
        metrics = {
            source: binary_metrics(
                base.labels[np.asarray([value == source for value in base.sources])],
                probabilities[np.asarray([value == source for value in base.sources])],
            )
            for source in SOURCES
        }
    loso, loso_sources = _loso_metrics(
        dataset, candidate, epochs=epochs, seed=seed, runtime=runtime
    )
    if (
        not np.isfinite(probabilities).all()
        or covered != set(SOURCES)
        or not 0.0 <= fraction < 0.10
        or set(loso) != set(SOURCES)
    ):
        raise RuntimeError("literature-grounded V9 evaluation is incomplete")
    return LiteratureGroundedEvaluationV9(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        loso_metrics=loso,
        loso_train_sources=loso_sources,
        model_fits=len(folds) + len(SOURCES),
        shared_gradient_sources=SOURCES,
        task_specific_parameter_fraction=float(fraction),
    )


def candidate_is_non_degrading(
    candidate_metrics: dict[str, dict[str, float]],
    comparator_metrics: dict[str, dict[str, float]],
) -> bool:
    if (
        type(candidate_metrics) is not dict
        or type(comparator_metrics) is not dict
        or set(candidate_metrics) != set(SOURCES)
        or set(comparator_metrics) != set(SOURCES)
    ):
        raise ValueError("non-degradation requires exact source metrics")
    return all(
        float(candidate_metrics[source]["accuracy"]) + 0.01
        >= float(comparator_metrics[source]["accuracy"])
        and float(candidate_metrics[source]["auroc"]) + 0.01
        >= float(comparator_metrics[source]["auroc"])
        and float(candidate_metrics[source]["sensitivity"]) >= 0.85
        for source in SOURCES
    )


def screen_candidate_ids(
    individual_metrics: dict[str, dict[str, dict[str, float]]],
) -> tuple[str, ...]:
    if type(individual_metrics) is not dict or set(individual_metrics) != {
        "LGS9-000", "LGS9-001", "LGS9-002"
    }:
        raise ValueError("screen authorization requires the three individual models")
    selected = ["LGS9-000", "LGS9-001", "LGS9-002"]
    comparator = individual_metrics["LGS9-000"]
    if all(
        candidate_is_non_degrading(individual_metrics[candidate_id], comparator)
        for candidate_id in ("LGS9-001", "LGS9-002")
    ):
        selected.append("LGS9-003")
    return tuple(selected)


__all__ = [
    "LiteratureGroundedEvaluationV9",
    "candidate_is_non_degrading",
    "evaluate_literature_grounded_candidate",
    "screen_candidate_ids",
]
