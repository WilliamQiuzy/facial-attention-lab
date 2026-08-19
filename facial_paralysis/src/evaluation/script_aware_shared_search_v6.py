"""Participant-disjoint evaluation for the script-aware shared router v6."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2, _model_inputs, _scaled, _tensor,
)
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES, fit_clinical_scaler, participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.script_aware_shared_router_v6 import (
    ScriptAwareCandidateV6, ScriptAwareSharedRouterV6, candidate_registry_v6,
)


UNIVERSAL_AUXILIARY_WEIGHT = 0.5
_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


@dataclass(frozen=True)
class ScriptAwareEvaluationV6:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    threshold: float
    shared_gradient_sources: tuple[str, ...]
    gradient_cosines: dict[str, float]


def _gradient_audit(model, inputs, labels, tasks, sources):
    gradients = {}
    covered = set()
    action_mask = inputs[-2]
    for source in SOURCES:
        local = torch.tensor(
            [index for index, value in enumerate(sources) if value == source],
            dtype=torch.long, device=labels.device,
        )
        model.zero_grad(set_to_none=True)
        source_inputs = tuple(value.index_select(0, local) for value in inputs)
        tokens = model.shared_action_tokens(*source_inputs)
        local_tasks = tasks.index_select(0, local)
        endpoint = model.endpoint_embedding(tokens, action_mask.index_select(0, local), local_tasks)
        universal = model.universal_embedding(tokens, action_mask.index_select(0, local))
        local_labels = labels.index_select(0, local)
        routed = (1 - model.candidate.universal_blend) * model.task_logits_from_embedding(endpoint, local_tasks)
        routed = routed + model.candidate.universal_blend * model.universal_head(universal).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(routed, local_labels)
        loss.backward()
        clinical = model.backbone.clinical_encoder[0].weight.grad
        patient = model.backbone.patient_projection.weight.grad
        if clinical is None or patient is None or float(clinical.norm()) <= 0 or float(patient.norm()) <= 0:
            raise RuntimeError("a source failed to update the shared v6 trunk")
        covered.add(source)
        gradients[source] = patient.detach().flatten().cpu()
    cosines = {}
    for first, second in combinations(SOURCES, 2):
        cosines[f"{first}__{second}"] = float(F.cosine_similarity(
            gradients[first][None], gradients[second][None]
        ).item())
    return tuple(source for source in SOURCES if source in covered), cosines


def evaluate_script_aware_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: ScriptAwareCandidateV6,
    *, epochs: int, n_splits: int = 6, seed: int = 0, device: str = "cpu",
) -> ScriptAwareEvaluationV6:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not ScriptAwareCandidateV6
        or candidate not in candidate_registry_v6()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid script-aware evaluation configuration")
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    audited_sources = None
    gradient_cosines = None
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda": torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = ScriptAwareSharedRouterV6(candidate).to(runtime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        train_sources = tuple(base.sources[index] for index in train)
        weights = _tensor(source_class_balanced_weights(
            base.labels[train], train_sources
        ).astype(np.float32), runtime)
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        inputs = _model_inputs(dataset, original, mirrored, train, runtime)
        if fold_index == 0:
            audited_sources, gradient_cosines = _gradient_audit(
                model, inputs, labels, tasks, train_sources
            )
        for _ in range(epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            tokens = model.shared_action_tokens(*inputs)
            endpoint = model.endpoint_embedding(tokens, inputs[-2], tasks)
            universal = model.universal_embedding(tokens, inputs[-2])
            task_logits = model.task_logits_from_embedding(endpoint, tasks)
            universal_logits = model.universal_head(universal).squeeze(-1)
            blend = candidate.universal_blend
            routed = (1 - blend) * task_logits + blend * universal_logits
            losses = F.binary_cross_entropy_with_logits(routed, labels, reduction="none")
            losses = losses + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
                universal_logits, labels, reduction="none"
            )
            torch.sum(losses * weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
            held_tasks = _tensor(np.asarray([
                _SOURCE_TASK_CODE[base.sources[index]] for index in held
            ], dtype=np.int64), runtime)
            tokens = model.shared_action_tokens(*held_inputs)
            probabilities[held] = torch.sigmoid(
                model.routed_logits(tokens, held_inputs[-2], held_tasks)
            ).cpu().numpy()
        del model, optimizer
    if not np.isfinite(probabilities).all() or audited_sources is None or gradient_cosines is None:
        raise RuntimeError("script-aware evaluation did not cover all participants")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([value == source for value in base.sources])
        metrics[source] = binary_metrics(base.labels[selected], probabilities[selected])
    return ScriptAwareEvaluationV6(
        probabilities=_immutable(probabilities), metrics=metrics,
        model_fits=len(folds), threshold=0.5,
        shared_gradient_sources=audited_sources, gradient_cosines=gradient_cosines,
    )


def rank_script_aware_results(results: dict[str, ScriptAwareEvaluationV6]) -> tuple[str, ...]:
    expected = {item.candidate_id for item in candidate_registry_v6()}
    if type(results) is not dict or set(results) != expected:
        raise ValueError("ranking requires every frozen v6 candidate")
    def key(item):
        metrics = results[item].metrics
        balanced = [float(metrics[source]["balanced_accuracy"]) for source in SOURCES]
        specificity = [float(metrics[source]["specificity"]) for source in SOURCES]
        aurocs = [float(metrics[source]["auroc"]) for source in SOURCES]
        return (-min(balanced), -min(specificity), -min(aurocs), -float(np.mean(balanced)), item)
    return tuple(sorted(expected, key=key))


__all__ = ["ScriptAwareEvaluationV6", "evaluate_script_aware_candidate", "rank_script_aware_results"]
