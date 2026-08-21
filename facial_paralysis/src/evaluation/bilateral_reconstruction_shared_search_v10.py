"""Closed participant-disjoint V10 evaluation based on BLV9-009."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.broad_literature_shared_search_v9 import (
    candidate_is_promotable,
    evaluate_broad_literature_candidate,
)
from src.evaluation.distilled_shared_search_v9 import configure_deterministic_training_v9
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
from src.models.bilateral_reconstruction_candidate_registry_v10 import (
    BilateralReconstructionCandidateV10,
    candidate_registry_v10,
)
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.training.bilateral_masked_reconstruction_v10 import (
    BilateralMaskedReconstructionV10,
)
from src.training.broad_literature_objectives_v9 import (
    SharpnessAwareControllerV9,
    classification_objective,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _v9_candidate():
    matched = tuple(
        row for row in candidate_registry_v9()
        if row.mechanism == "masked_clinical_reconstruction"
    )
    if len(matched) != 1:
        raise RuntimeError("the BLV9-009 research baseline is unavailable")
    return matched[0]


def uses_sam(candidate: BilateralReconstructionCandidateV10) -> bool:
    if (
        type(candidate) is not BilateralReconstructionCandidateV10
        or candidate not in candidate_registry_v10()
    ):
        raise ValueError("SAM query requires one frozen V10 candidate")
    return candidate.optimizer_mode == "sam"


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class BilateralReconstructionEvaluationV10:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    loso_metrics: dict[str, dict[str, float]]
    loso_train_sources: tuple[tuple[str, tuple[str, ...]], ...]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    task_specific_parameter_fraction: float
    active_candidate_id: str


def _source_counts(labels: torch.Tensor, task_codes: torch.Tensor) -> torch.Tensor:
    counts = torch.zeros(3, 2, dtype=torch.long, device=labels.device)
    for source in range(3):
        for label in range(2):
            counts[source, label] = torch.sum(
                (task_codes == source) & (labels.to(torch.long) == label)
            )
    return counts


def _logits(
    model: BroadLiteratureSharedRouterV9,
    inputs: tuple[torch.Tensor, ...],
    task_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.shared_action_tokens(*inputs)
    return model.routed_and_universal_logits(tokens, inputs[-2], task_codes)


def _objective(
    model: BroadLiteratureSharedRouterV9,
    auxiliary: BilateralMaskedReconstructionV10,
    inputs: tuple[torch.Tensor, ...],
    tasks: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    class_counts: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    routed, universal = _logits(model, inputs, tasks)
    return classification_objective(
        _v9_candidate(), routed, universal, labels, weights, tasks, class_counts
    ) + auxiliary.loss(model, inputs, seed=seed)


def _fit_model(
    dataset: MedicalSharedDatasetV2,
    candidate: BilateralReconstructionCandidateV10,
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
    model = BroadLiteratureSharedRouterV9(_v9_candidate()).to(runtime)
    auxiliary = BilateralMaskedReconstructionV10(candidate).to(runtime)
    parameters = tuple(model.parameters()) + tuple(auxiliary.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-3)
    controller = (
        SharpnessAwareControllerV9(parameters, rho=0.05, adaptive=False)
        if uses_sam(candidate) else None
    )
    train_sources = tuple(base.sources[index] for index in train)
    weights = _tensor(source_class_balanced_weights(
        base.labels[train], train_sources
    ).astype(np.float32), runtime)
    labels = _tensor(base.labels[train].astype(np.float32), runtime)
    tasks = _tensor(np.asarray([
        _SOURCE_TASK_CODE[source] for source in train_sources
    ], dtype=np.int64), runtime)
    class_counts = _source_counts(labels, tasks)
    inputs = _model_inputs(dataset, original, mirrored, train, runtime)

    covered: set[str] = set()
    if audit_sources:
        for source in SOURCES:
            local = torch.tensor([
                index for index, observed in enumerate(train_sources)
                if observed == source
            ], dtype=torch.long, device=runtime)
            model.zero_grad(set_to_none=True)
            selected_inputs = tuple(value.index_select(0, local) for value in inputs)
            routed, _ = _logits(model, selected_inputs, tasks.index_select(0, local))
            F.binary_cross_entropy_with_logits(
                routed, labels.index_select(0, local)
            ).backward()
            clinical = model.base.base.backbone.clinical_encoder[0].weight.grad
            patient = model.base.base.backbone.patient_projection.weight.grad
            if (
                clinical is None or patient is None
                or float(clinical.norm()) <= 0.0 or float(patient.norm()) <= 0.0
            ):
                raise RuntimeError("a source failed the shared-gradient audit")
            covered.add(source)

    for epoch in range(epochs):
        model.train()
        auxiliary.train()
        optimizer.zero_grad(set_to_none=True)
        objective = _objective(
            model, auxiliary, inputs, tasks, labels, weights, class_counts,
            seed=local_seed * 1000 + epoch,
        )
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("V10 produced a nonfinite objective")
        objective.backward()
        if controller is None:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        else:
            controller.first_step()
            optimizer.zero_grad(set_to_none=True)
            second = _objective(
                model, auxiliary, inputs, tasks, labels, weights, class_counts,
                seed=local_seed * 1000 + epoch,
            )
            second.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            controller.second_step(optimizer)
    return model, original, mirrored, covered


def _predict_routed(
    model: BroadLiteratureSharedRouterV9,
    dataset: MedicalSharedDatasetV2,
    original: np.ndarray,
    mirrored: np.ndarray,
    held: np.ndarray,
    runtime: torch.device,
) -> np.ndarray:
    inputs = _model_inputs(dataset, original, mirrored, held, runtime)
    tasks = _tensor(np.asarray([
        _SOURCE_TASK_CODE[dataset.base.sources[index]] for index in held
    ], dtype=np.int64), runtime)
    model.eval()
    with torch.no_grad():
        routed, _ = _logits(model, inputs, tasks)
    return torch.sigmoid(routed).cpu().numpy().astype(np.float64)


def _loso_metrics(
    dataset: MedicalSharedDatasetV2,
    candidate: BilateralReconstructionCandidateV10,
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
        inputs = _model_inputs(dataset, original, mirrored, held, runtime)
        model.eval()
        with torch.no_grad():
            tokens = model.shared_action_tokens(*inputs)
            probabilities = torch.sigmoid(
                model.universal_logits(tokens, inputs[-2])
            ).cpu().numpy()
        results[target_source] = binary_metrics(base.labels[held], probabilities)
        training_sources.append((target_source, tuple(
            source for source in SOURCES if source != target_source
        )))
        del model
    return results, tuple(training_sources)


def evaluate_bilateral_reconstruction_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: BilateralReconstructionCandidateV10,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> BilateralReconstructionEvaluationV10:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not BilateralReconstructionCandidateV10
        or candidate not in candidate_registry_v10()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid bilateral reconstruction V10 evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested V10 runtime is unavailable")
    configure_deterministic_training_v9(runtime)

    if candidate.candidate_id == "BRV10-000":
        baseline = evaluate_broad_literature_candidate(
            dataset, _v9_candidate(), epochs=epochs, n_splits=n_splits,
            seed=seed, device=device,
        )
        return BilateralReconstructionEvaluationV10(
            probabilities=baseline.probabilities,
            metrics=baseline.metrics,
            loso_metrics=baseline.loso_metrics,
            loso_train_sources=baseline.loso_train_sources,
            model_fits=baseline.model_fits,
            shared_gradient_sources=baseline.shared_gradient_sources,
            task_specific_parameter_fraction=baseline.task_specific_parameter_fraction,
            active_candidate_id=candidate.candidate_id,
        )

    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    fraction = np.nan
    for fold_index, (train, held) in enumerate(folds):
        model, original, mirrored, audited = _fit_model(
            dataset, candidate, train, epochs=epochs,
            local_seed=seed * 1009 + fold_index, runtime=runtime,
            audit_sources=fold_index == 0,
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
        or not 0.0 <= float(fraction) < 0.10
        or set(loso) != set(SOURCES)
    ):
        raise RuntimeError("bilateral reconstruction V10 evaluation is incomplete")
    return BilateralReconstructionEvaluationV10(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        loso_metrics=loso,
        loso_train_sources=loso_sources,
        model_fits=len(folds) + len(SOURCES),
        shared_gradient_sources=SOURCES,
        task_specific_parameter_fraction=float(fraction),
        active_candidate_id=candidate.candidate_id,
    )


__all__ = [
    "BilateralReconstructionEvaluationV10",
    "candidate_is_promotable",
    "evaluate_bilateral_reconstruction_candidate",
    "uses_sam",
]
