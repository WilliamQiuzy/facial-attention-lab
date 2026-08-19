"""Fold-local response-statistic PCA and shared-router evaluation v7."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from src.evaluation.medically_gated_shared_search_v2 import MedicalSharedDatasetV2, _scaled, _tensor
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES, fit_clinical_scaler, participant_disjoint_folds, source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.response_statistic_shared_router_v7 import (
    ResponseStatisticCandidateV7, ResponseStatisticSharedRouterV7, candidate_registry_v7,
)
from src.preprocessing.shared_response_statistics_v7 import dense_response_statistics_v7


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values):
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _fold_dense_pca(statistics, available, train, output_dim, seed):
    training_rows = statistics[train][available[train]]
    if training_rows.shape[0] < 2:
        raise ValueError("fold-local dense PCA requires at least two action rows")
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(training_rows)
    components = min(output_dim, scaled_train.shape[0] - 1, scaled_train.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(scaled_train)
    output = np.zeros(available.shape + (output_dim,), dtype=np.float32)
    all_rows = statistics[available]
    transformed = pca.transform(scaler.transform(all_rows)).astype(np.float32)
    output[available, :components] = transformed
    if not np.isfinite(output).all() or np.any(output[~available] != 0.0):
        raise RuntimeError("fold-local response PCA failed finite missingness QC")
    return output


@dataclass(frozen=True)
class ResponseStatisticEvaluationV7:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    threshold: float
    shared_gradient_sources: tuple[str, ...]


def evaluate_response_statistic_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: ResponseStatisticCandidateV7,
    *, epochs: int, n_splits: int = 6, seed: int = 0, device: str = "cpu",
) -> ResponseStatisticEvaluationV7:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not ResponseStatisticCandidateV7
        or candidate not in candidate_registry_v7()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid response-statistic evaluation configuration")
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    base = dataset.base
    statistics = dense_response_statistics_v7(
        base.dense_original, base.dense_mirrored,
        dataset.dense_timestamps, base.dense_available,
    )
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered = set()
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda": torch.cuda.manual_seed_all(local_seed)
        clinical_scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, clinical_scaler.mean, clinical_scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, clinical_scaler.mean, clinical_scaler.scale)
        dense_pca = _fold_dense_pca(
            statistics, base.dense_available, train, candidate.pca_dim, local_seed
        )
        model = ResponseStatisticSharedRouterV7(candidate).to(runtime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        train_sources = tuple(base.sources[index] for index in train)
        weights = _tensor(source_class_balanced_weights(
            base.labels[train], train_sources
        ).astype(np.float32), runtime)
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        inputs = (
            _tensor(original[train], runtime), _tensor(mirrored[train], runtime),
            _tensor(dense_pca[train], runtime), _tensor(base.dense_available[train], runtime),
            _tensor(base.action_mask[train], runtime), _tensor(base.action_codes[train], runtime),
        )
        if fold_index == 0:
            for source in SOURCES:
                local = torch.tensor([
                    index for index, value in enumerate(train_sources) if value == source
                ], dtype=torch.long, device=runtime)
                model.zero_grad(set_to_none=True)
                local_inputs = tuple(value.index_select(0, local) for value in inputs)
                tokens = model.shared_action_tokens(*local_inputs)
                loss = F.binary_cross_entropy_with_logits(
                    model.routed_logits(tokens, local_inputs[-2], tasks.index_select(0, local)),
                    labels.index_select(0, local),
                )
                loss.backward()
                if model.clinical_encoder[0].weight.grad is None or model.patient_projection.weight.grad is None:
                    raise RuntimeError("a source failed the v7 shared-gradient audit")
                covered.add(source)
        for _ in range(epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            tokens = model.shared_action_tokens(*inputs)
            endpoint = model._patient(tokens, inputs[-2], tasks)
            universal = model._patient(tokens, inputs[-2])
            task_logits = model.task_logits_from_embedding(endpoint, tasks)
            universal_logits = model.universal_head(universal).squeeze(-1)
            routed = 0.75 * task_logits + 0.25 * universal_logits
            losses = F.binary_cross_entropy_with_logits(routed, labels, reduction="none")
            losses = losses + 0.5 * F.binary_cross_entropy_with_logits(
                universal_logits, labels, reduction="none"
            )
            torch.sum(losses * weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        held_tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[base.sources[index]] for index in held
        ], dtype=np.int64), runtime)
        held_inputs = (
            _tensor(original[held], runtime), _tensor(mirrored[held], runtime),
            _tensor(dense_pca[held], runtime), _tensor(base.dense_available[held], runtime),
            _tensor(base.action_mask[held], runtime), _tensor(base.action_codes[held], runtime),
        )
        with torch.no_grad():
            tokens = model.shared_action_tokens(*held_inputs)
            probabilities[held] = torch.sigmoid(
                model.routed_logits(tokens, held_inputs[-2], held_tasks)
            ).cpu().numpy()
        del model, optimizer
    if not np.isfinite(probabilities).all() or covered != set(SOURCES):
        raise RuntimeError("response-statistic evaluation did not cover all participants")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([value == source for value in base.sources])
        metrics[source] = binary_metrics(base.labels[selected], probabilities[selected])
    return ResponseStatisticEvaluationV7(
        probabilities=_immutable(probabilities), metrics=metrics, model_fits=len(folds),
        threshold=0.5, shared_gradient_sources=SOURCES,
    )


def rank_response_statistic_results(results: dict[str, ResponseStatisticEvaluationV7]):
    expected = {item.candidate_id for item in candidate_registry_v7()}
    if type(results) is not dict or set(results) != expected: raise ValueError("v7 ranking requires all candidates")
    def key(item):
        metrics=results[item].metrics
        b=[metrics[source]["balanced_accuracy"] for source in SOURCES]
        s=[metrics[source]["specificity"] for source in SOURCES]
        a=[metrics[source]["auroc"] for source in SOURCES]
        return (-min(b),-min(s),-min(a),-float(np.mean(b)),item)
    return tuple(sorted(expected,key=key))


__all__ = ["ResponseStatisticEvaluationV7", "evaluate_response_statistic_candidate", "rank_response_statistic_results"]
