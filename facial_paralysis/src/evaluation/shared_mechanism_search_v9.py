"""Participant-disjoint evaluator for the low-sample shared mechanism V9."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.distilled_shared_search_v9 import (
    _clinical_features,
    _dense_features,
    configure_deterministic_training_v9,
)
from src.evaluation.medically_gated_shared_search_v2 import MedicalSharedDatasetV2
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.shared_mechanism_router_v9 import (
    SharedMechanismCandidateV9,
    SharedMechanismRouterV9,
    candidate_registry_v9,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class MechanismActionEvidenceV9:
    values: np.ndarray
    action_mask: np.ndarray
    dense_available: np.ndarray


def mechanism_action_tensor(
    dataset: MedicalSharedDatasetV2,
) -> MechanismActionEvidenceV9:
    if type(dataset) is not MedicalSharedDatasetV2:
        raise ValueError("mechanism action tensor requires a validated dataset")
    base = dataset.base
    count = len(base.labels)
    clinical = _clinical_features(dataset).reshape(count, 13, 220)
    dense = _dense_features(dataset).reshape(count, 13, 16)
    action_mask = np.zeros((count, 13), dtype=bool)
    dense_available = np.zeros((count, 13), dtype=bool)
    for participant in range(count):
        for code in range(13):
            selected = base.action_mask[participant] & (
                base.action_codes[participant] == code
            )
            action_mask[participant, code] = bool(selected.any())
            dense_available[participant, code] = bool(
                (selected & base.dense_available[participant]).any()
            )
    values = np.concatenate((
        clinical,
        dense,
        dense_available[..., None].astype(np.float64),
    ), axis=-1)
    if not np.isfinite(values).all() or np.any(dense_available & ~action_mask):
        raise RuntimeError("mechanism action construction failed QC")
    return MechanismActionEvidenceV9(
        values=_immutable(values),
        action_mask=_immutable(action_mask),
        dense_available=_immutable(dense_available),
    )


def _scaled_evidence(
    evidence: MechanismActionEvidenceV9,
    training_indices: np.ndarray,
) -> np.ndarray:
    values = np.array(evidence.values, copy=True)
    active = evidence.action_mask[training_indices]
    clinical_rows = values[training_indices, :, :220][active]
    clinical_mean = clinical_rows.mean(axis=0)
    clinical_scale = clinical_rows.std(axis=0)
    clinical_scale[clinical_scale < 1e-8] = 1.0
    values[..., :220] = (values[..., :220] - clinical_mean) / clinical_scale
    dense_active = evidence.dense_available[training_indices]
    if dense_active.any():
        dense_rows = values[training_indices, :, 220:236][dense_active]
        dense_mean = dense_rows.mean(axis=0)
        dense_scale = dense_rows.std(axis=0)
        dense_scale[dense_scale < 1e-8] = 1.0
        values[..., 220:236] = (values[..., 220:236] - dense_mean) / dense_scale
        values[..., 220:236][~evidence.dense_available] = 0.0
    values[~evidence.action_mask] = 0.0
    if not np.isfinite(values).all():
        raise RuntimeError("fold-local mechanism scaling produced nonfinite values")
    return values.astype(np.float32)


@dataclass(frozen=True)
class MechanismEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    task_specific_parameter_fraction: float


def evaluate_mechanism_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: SharedMechanismCandidateV9,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> MechanismEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not SharedMechanismCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid shared mechanism evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("shared mechanism runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    base = dataset.base
    evidence = mechanism_action_tensor(dataset)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    fraction = np.nan
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        values = _scaled_evidence(evidence, train)
        model = SharedMechanismRouterV9(candidate).to(runtime)
        fraction = model.task_specific_parameter_fraction()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        train_sources = tuple(base.sources[index] for index in train)
        weights = torch.from_numpy(
            np.array(source_class_balanced_weights(base.labels[train], train_sources),
                     dtype=np.float32, copy=True)
        ).to(runtime)
        labels = torch.from_numpy(base.labels[train].astype(np.float32, copy=True)).to(
            runtime
        )
        tasks = torch.tensor([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=torch.long, device=runtime)
        x_train = torch.from_numpy(np.array(values[train], copy=True)).to(runtime)
        mask_train = torch.from_numpy(
            np.array(evidence.action_mask[train], copy=True)
        ).to(runtime)

        if fold_index == 0:
            for source in SOURCES:
                local = torch.tensor([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=torch.long, device=runtime)
                model.zero_grad(set_to_none=True)
                routed, _ = model(
                    x_train.index_select(0, local),
                    mask_train.index_select(0, local),
                    tasks.index_select(0, local),
                )
                F.binary_cross_entropy_with_logits(
                    routed, labels.index_select(0, local)
                ).backward()
                first = model.action_encoder[0].weight.grad
                patient = model.patient_projection.weight.grad
                if (
                    first is None or patient is None
                    or float(first.norm()) <= 0.0 or float(patient.norm()) <= 0.0
                ):
                    raise RuntimeError("a source failed shared mechanism gradients")
                covered.add(source)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            routed, universal = model(x_train, mask_train, tasks)
            losses = F.binary_cross_entropy_with_logits(
                routed, labels, reduction="none"
            ) + 0.5 * F.binary_cross_entropy_with_logits(
                universal, labels, reduction="none"
            )
            torch.sum(losses * weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        x_held = torch.from_numpy(np.array(values[held], copy=True)).to(runtime)
        mask_held = torch.from_numpy(
            np.array(evidence.action_mask[held], copy=True)
        ).to(runtime)
        held_tasks = torch.tensor([
            _SOURCE_TASK_CODE[base.sources[index]] for index in held
        ], dtype=torch.long, device=runtime)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_held, mask_held, held_tasks)
            probabilities[held] = torch.sigmoid(logits).cpu().numpy()
        del model, optimizer
    if (
        not np.isfinite(probabilities).all()
        or covered != set(SOURCES)
        or not 0 <= fraction < 0.10
    ):
        raise RuntimeError("shared mechanism evaluation failed its boundary")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in base.sources])
        metrics[source] = binary_metrics(base.labels[selected], probabilities[selected])
    return MechanismEvaluationV9(
        probabilities=_immutable(probabilities), metrics=metrics,
        model_fits=len(folds), shared_gradient_sources=SOURCES,
        task_specific_parameter_fraction=float(fraction),
    )


def rank_mechanism_results(
    results: dict[str, MechanismEvaluationV9],
) -> tuple[str, ...]:
    expected = {row.candidate_id for row in candidate_registry_v9()}
    if type(results) is not dict or set(results) != expected:
        raise ValueError("shared mechanism ranking requires the complete registry")

    def key(candidate_id: str):
        metrics = results[candidate_id].metrics
        feasible = all(metrics[source]["sensitivity"] >= 0.85 for source in SOURCES)
        return (
            not feasible,
            -min(metrics[source]["specificity"] for source in SOURCES),
            -min(metrics[source]["auroc"] for source in SOURCES),
            -min(metrics[source]["accuracy"] for source in SOURCES),
            -float(np.mean([metrics[source]["accuracy"] for source in SOURCES])),
            candidate_id,
        )
    return tuple(sorted(expected, key=key))


__all__ = [
    "MechanismActionEvidenceV9", "MechanismEvaluationV9",
    "evaluate_mechanism_candidate", "mechanism_action_tensor",
    "rank_mechanism_results",
]
