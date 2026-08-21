"""Participant-disjoint evaluation for twenty isolated shared V9 mechanisms."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.distilled_shared_search_v9 import configure_deterministic_training_v9
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
from src.models.broad_literature_candidate_registry_v9 import (
    BroadLiteratureCandidateV9,
    candidate_registry_v9,
)
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.models.residual_shared_router_v8 import candidate_registry_v8
from src.training.broad_literature_objectives_v9 import (
    RepresentationAuxiliariesV9,
    SWAAccumulatorV9,
    SharpnessAwareControllerV9,
    action_dropout_mask,
    classification_objective,
    modality_dropout_mask,
    symmetric_binary_kl,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_SELF_SUPERVISION = {
    "cross_view_vicreg",
    "cross_view_barlow_twins",
    "masked_clinical_reconstruction",
    "masked_action_reconstruction",
    "clinical_to_dense_reconstruction",
}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class BroadLiteratureEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    loso_metrics: dict[str, dict[str, float]]
    loso_train_sources: tuple[tuple[str, tuple[str, ...]], ...]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    task_specific_parameter_fraction: float
    active_mechanism: str


def _candidate_setting(candidate: BroadLiteratureCandidateV9, name: str):
    values = dict(candidate.settings)
    if name not in values:
        raise ValueError(f"candidate does not define {name}")
    return values[name]


def _source_counts(
    labels: torch.Tensor,
    task_codes: torch.Tensor,
) -> torch.Tensor:
    counts = torch.zeros(3, 2, dtype=torch.long, device=labels.device)
    for source in range(3):
        for label in range(2):
            counts[source, label] = torch.sum(
                (task_codes == source) & (labels.to(torch.long) == label)
            )
    return counts


def _drop_dense_inputs(
    inputs: tuple[torch.Tensor, ...],
    probability: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    changed = list(inputs)
    changed[5] = modality_dropout_mask(
        inputs[5], probability=probability, seed=seed
    )
    changed[4] = inputs[4] & changed[5].unsqueeze(-1)
    return tuple(changed)


def _drop_action_inputs(
    inputs: tuple[torch.Tensor, ...],
    probability: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    changed = list(inputs)
    changed[7] = action_dropout_mask(
        inputs[7], probability=probability, seed=seed
    )
    changed[5] = inputs[5] & changed[7]
    changed[4] = inputs[4] & changed[5].unsqueeze(-1)
    return tuple(changed)


def _logits(
    model: BroadLiteratureSharedRouterV9,
    inputs: tuple[torch.Tensor, ...],
    task_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.shared_action_tokens(*inputs)
    return model.routed_and_universal_logits(tokens, inputs[-2], task_codes)


def _epoch_objective(
    model: BroadLiteratureSharedRouterV9,
    auxiliary: RepresentationAuxiliariesV9 | None,
    candidate: BroadLiteratureCandidateV9,
    inputs: tuple[torch.Tensor, ...],
    task_codes: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    class_counts: torch.Tensor,
    *,
    epoch_seed: int,
) -> torch.Tensor:
    mechanism = candidate.mechanism
    if mechanism == "r_drop":
        first_routed, first_universal = _logits(model, inputs, task_codes)
        second_routed, second_universal = _logits(model, inputs, task_codes)
        first = classification_objective(
            candidate, first_routed, first_universal, labels, weights,
            task_codes, class_counts,
        )
        second = classification_objective(
            candidate, second_routed, second_universal, labels, weights,
            task_codes, class_counts,
        )
        return 0.5 * (first + second) + float(
            _candidate_setting(candidate, "symmetric_kl_weight")
        ) * symmetric_binary_kl(first_routed, second_routed)
    if mechanism == "modality_dropout":
        observed_inputs = _drop_dense_inputs(
            inputs, float(_candidate_setting(candidate, "dense_drop_probability")),
            epoch_seed,
        )
        routed, universal = _logits(model, observed_inputs, task_codes)
        return classification_objective(
            candidate, routed, universal, labels, weights, task_codes, class_counts
        )
    if mechanism == "action_dropout_consistency":
        dropped_inputs = _drop_action_inputs(
            inputs, float(_candidate_setting(candidate, "action_drop_probability")),
            epoch_seed,
        )
        full_routed, full_universal = _logits(model, inputs, task_codes)
        drop_routed, drop_universal = _logits(model, dropped_inputs, task_codes)
        full = classification_objective(
            candidate, full_routed, full_universal, labels, weights,
            task_codes, class_counts,
        )
        dropped = classification_objective(
            candidate, drop_routed, drop_universal, labels, weights,
            task_codes, class_counts,
        )
        return 0.5 * (full + dropped) + float(
            _candidate_setting(candidate, "consistency_weight")
        ) * symmetric_binary_kl(full_routed, drop_routed)
    routed, universal = _logits(model, inputs, task_codes)
    objective = classification_objective(
        candidate, routed, universal, labels, weights, task_codes, class_counts
    )
    if auxiliary is not None:
        objective = objective + auxiliary.loss(
            model, inputs, task_codes, seed=epoch_seed
        )
    return objective


def _fit_model(
    dataset: MedicalSharedDatasetV2,
    candidate: BroadLiteratureCandidateV9,
    train: np.ndarray,
    *,
    epochs: int,
    local_seed: int,
    runtime: torch.device,
    audit_sources: bool,
):
    if candidate.mechanism == "swa" and epochs < 20:
        raise ValueError("the frozen SWA candidate requires all twenty epochs")
    base = dataset.base
    torch.manual_seed(local_seed)
    if runtime.type == "cuda":
        torch.cuda.manual_seed_all(local_seed)
    scaler = fit_clinical_scaler(base, train)
    original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
    mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
    model = BroadLiteratureSharedRouterV9(candidate).to(runtime)
    auxiliary = (
        RepresentationAuxiliariesV9(candidate).to(runtime)
        if candidate.mechanism in _SELF_SUPERVISION else None
    )
    parameters = tuple(model.parameters()) + (
        tuple(auxiliary.parameters()) if auxiliary is not None else ()
    )
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-3)
    controller = None
    if candidate.mechanism in {"sam", "asam"}:
        controller = SharpnessAwareControllerV9(
            parameters,
            rho=float(_candidate_setting(candidate, "rho")),
            adaptive=candidate.mechanism == "asam",
            eta=float(dict(candidate.settings).get("eta", 0.01)),
        )
    swa = SWAAccumulatorV9(model) if candidate.mechanism == "swa" else None
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
            local_inputs = tuple(value.index_select(0, local) for value in inputs)
            routed, _ = _logits(model, local_inputs, tasks.index_select(0, local))
            F.binary_cross_entropy_with_logits(
                routed, labels.index_select(0, local)
            ).backward()
            clinical = model.base.base.backbone.clinical_encoder[0].weight.grad
            patient = model.base.base.backbone.patient_projection.weight.grad
            if (
                clinical is None or patient is None
                or float(clinical.norm()) <= 0.0 or float(patient.norm()) <= 0.0
            ):
                raise RuntimeError("a source failed the genuinely shared gradient audit")
            covered.add(source)

    for epoch in range(epochs):
        model.train()
        if auxiliary is not None:
            auxiliary.train()
        optimizer.zero_grad(set_to_none=True)
        objective = _epoch_objective(
            model, auxiliary, candidate, inputs, tasks, labels, weights,
            class_counts, epoch_seed=local_seed * 1000 + epoch,
        )
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("broad V9 produced a nonfinite objective")
        objective.backward()
        if controller is None:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        else:
            controller.first_step()
            optimizer.zero_grad(set_to_none=True)
            second = _epoch_objective(
                model, auxiliary, candidate, inputs, tasks, labels, weights,
                class_counts, epoch_seed=local_seed * 1000 + epoch,
            )
            second.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            controller.second_step(optimizer)
        if swa is not None and 11 <= epoch + 1 <= 20:
            swa.update(model)
    if swa is not None:
        swa.copy_to(model)
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
    candidate: BroadLiteratureCandidateV9,
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


def evaluate_broad_literature_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: BroadLiteratureCandidateV9,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> BroadLiteratureEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not BroadLiteratureCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid broad literature V9 evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested broad V9 runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    fraction = np.nan
    if candidate.mechanism == "exact_v8_comparator":
        baseline = evaluate_residual_candidate(
            dataset,
            next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"),
            epochs=epochs, n_splits=n_splits, seed=seed, device=device,
        )
        probabilities[:] = baseline.probabilities
        metrics = baseline.metrics
        covered.update(baseline.shared_gradient_sources)
        fraction = BroadLiteratureSharedRouterV9(candidate).task_specific_parameter_fraction()
    else:
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
        or not 0.0 <= float(fraction) < 0.10
        or set(loso) != set(SOURCES)
    ):
        raise RuntimeError("broad literature V9 evaluation is incomplete")
    return BroadLiteratureEvaluationV9(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        loso_metrics=loso,
        loso_train_sources=loso_sources,
        model_fits=len(folds) + len(SOURCES),
        shared_gradient_sources=SOURCES,
        task_specific_parameter_fraction=float(fraction),
        active_mechanism=candidate.mechanism,
    )


def candidate_is_promotable(
    observed: dict[str, dict[str, float]],
    comparator: dict[str, dict[str, float]],
) -> bool:
    if (
        type(observed) is not dict or type(comparator) is not dict
        or set(observed) != set(SOURCES) or set(comparator) != set(SOURCES)
    ):
        raise ValueError("promotion requires exact three-source metrics")
    if not all(
        float(observed[source]["accuracy"]) >= 0.90
        and float(observed[source]["specificity"]) >= 0.80
        and float(observed[source]["sensitivity"]) >= 0.85
        and float(observed[source]["auroc"]) >= 0.92
        and float(observed[source]["accuracy"]) + 0.01
        >= float(comparator[source]["accuracy"])
        and float(observed[source]["auroc"]) + 0.01
        >= float(comparator[source]["auroc"])
        for source in SOURCES
    ):
        return False
    return min(float(observed[source]["specificity"]) for source in SOURCES) > min(
        float(comparator[source]["specificity"]) for source in SOURCES
    )


__all__ = [
    "BroadLiteratureEvaluationV9",
    "candidate_is_promotable",
    "evaluate_broad_literature_candidate",
]
