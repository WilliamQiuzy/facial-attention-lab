"""Participant-disjoint evaluator for full-mesh shared action phenotypes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.distilled_shared_search_v9 import configure_deterministic_training_v9
from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2, _model_inputs, _scaled, _tensor,
)
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES, fit_clinical_scaler, participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.script_phenotype_router_v9 import (
    ScriptPhenotypeCandidateV9, ScriptPhenotypeRouterV9, candidate_registry_v9,
)
from src.models.residual_shared_router_v8 import candidate_registry_v8


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class PhenotypeEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    task_specific_parameter_fraction: float


def evaluate_phenotype_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: ScriptPhenotypeCandidateV9,
    *, epochs: int, n_splits: int = 6, seed: int = 0, device: str = "cpu",
) -> PhenotypeEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not ScriptPhenotypeCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid script phenotype evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("script phenotype runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    if candidate.script_blend == 0.0:
        baseline = evaluate_residual_candidate(
            dataset,
            next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"),
            epochs=epochs, n_splits=n_splits, seed=seed, device=device,
        )
        fraction = ScriptPhenotypeRouterV9(candidate).task_specific_parameter_fraction()
        return PhenotypeEvaluationV9(
            probabilities=baseline.probabilities,
            metrics=baseline.metrics,
            model_fits=baseline.model_fits,
            shared_gradient_sources=baseline.shared_gradient_sources,
            task_specific_parameter_fraction=fraction,
        )
    base = dataset.base
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    fraction = np.nan
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = ScriptPhenotypeRouterV9(candidate).to(runtime)
        fraction = model.task_specific_parameter_fraction()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        train_sources = tuple(base.sources[index] for index in train)
        weights = _tensor(
            source_class_balanced_weights(base.labels[train], train_sources).astype(
                np.float32
            ), runtime,
        )
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        inputs = _model_inputs(dataset, original, mirrored, train, runtime)

        if fold_index == 0:
            for source in SOURCES:
                local = torch.tensor([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=torch.long, device=runtime)
                model.zero_grad(set_to_none=True)
                local_inputs = tuple(value.index_select(0, local) for value in inputs)
                tokens = model.shared_action_tokens(*local_inputs)
                logits = model.routed_logits(
                    tokens, local_inputs[-2], local_inputs[-1],
                    tasks.index_select(0, local),
                )
                F.binary_cross_entropy_with_logits(
                    logits, labels.index_select(0, local)
                ).backward()
                clinical = model.base.base.backbone.clinical_encoder[0].weight.grad
                patient = model.base.base.backbone.patient_projection.weight.grad
                if (
                    clinical is None or patient is None
                    or float(clinical.norm()) <= 0.0 or float(patient.norm()) <= 0.0
                ):
                    raise RuntimeError("a source failed shared phenotype gradients")
                covered.add(source)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tokens = model.shared_action_tokens(*inputs)
            routed, universal = model.routed_and_universal_logits(
                tokens, inputs[-2], inputs[-1], tasks
            )
            losses = F.binary_cross_entropy_with_logits(
                routed, labels, reduction="none"
            ) + 0.5 * F.binary_cross_entropy_with_logits(
                universal, labels, reduction="none"
            )
            torch.sum(losses * weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
        held_tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[base.sources[index]] for index in held
        ], dtype=np.int64), runtime)
        model.eval()
        with torch.no_grad():
            tokens = model.shared_action_tokens(*held_inputs)
            logits = model.routed_logits(
                tokens, held_inputs[-2], held_inputs[-1], held_tasks
            )
            probabilities[held] = torch.sigmoid(logits).cpu().numpy()
        del model, optimizer
    if (
        not np.isfinite(probabilities).all() or covered != set(SOURCES)
        or not 0 <= fraction < 0.10
    ):
        raise RuntimeError("script phenotype evaluation failed sharing QC")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in base.sources])
        metrics[source] = binary_metrics(base.labels[selected], probabilities[selected])
    return PhenotypeEvaluationV9(
        probabilities=_immutable(probabilities), metrics=metrics,
        model_fits=len(folds), shared_gradient_sources=SOURCES,
        task_specific_parameter_fraction=float(fraction),
    )


def rank_phenotype_results(
    results: dict[str, PhenotypeEvaluationV9],
) -> tuple[str, ...]:
    expected = {row.candidate_id for row in candidate_registry_v9()}
    if type(results) is not dict or set(results) != expected:
        raise ValueError("script phenotype ranking requires the complete registry")
    comparator = results["SAP9-000"].metrics

    def key(candidate_id: str):
        metrics = results[candidate_id].metrics
        feasible = all(
            metrics[source]["sensitivity"] >= 0.85
            and metrics[source]["accuracy"] + 0.01 >= comparator[source]["accuracy"]
            and metrics[source]["auroc"] + 0.01 >= comparator[source]["auroc"]
            for source in SOURCES
        )
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
    "PhenotypeEvaluationV9", "evaluate_phenotype_candidate",
    "rank_phenotype_results",
]
