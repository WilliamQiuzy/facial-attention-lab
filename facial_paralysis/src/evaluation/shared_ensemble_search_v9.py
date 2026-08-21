"""Deterministic deep ensembles of genuinely shared V8 encoders for V9."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.residual_shared_search_v8 import ResidualEvaluationV8
from src.evaluation.shared_clinical_encoder_v1 import SOURCES
from src.evaluation.universal_orofacial_v1 import binary_metrics


_V8_IDS = (
    "RSR8-000", "RSR8-001", "RSR8-002", "RSR8-003", "RSR8-004", "RSR8-005"
)


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class EnsembleCandidateV9:
    candidate_id: str
    member_candidate_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    aggregation: str


@dataclass(frozen=True)
class EnsembleEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    member_models: int
    model_fits: int
    shared_gradient_sources: tuple[str, ...]


def ensemble_candidate_registry_v9() -> tuple[EnsembleCandidateV9, ...]:
    groups = (
        ("RSR8-001",),
        ("RSR8-000", "RSR8-002", "RSR8-004"),
        ("RSR8-001", "RSR8-003", "RSR8-005"),
        _V8_IDS,
    )
    rows = []
    index = 0
    for members in groups:
        for seeds in ((0,), (0, 1, 2)):
            for aggregation in ("probability_mean", "logit_mean"):
                rows.append(EnsembleCandidateV9(
                    candidate_id=f"SEN9-{index:03d}",
                    member_candidate_ids=members,
                    seeds=seeds,
                    aggregation=aggregation,
                ))
                index += 1
    if len(rows) != 16 or len(set(rows)) != 16:
        raise AssertionError("the shared V9 ensemble registry drifted")
    return tuple(rows)


def evaluate_ensemble_candidate(
    labels: np.ndarray,
    sources: tuple[str, ...],
    base_results: dict[tuple[str, int], ResidualEvaluationV8],
    candidate: EnsembleCandidateV9,
) -> EnsembleEvaluationV9:
    labels = np.asarray(labels)
    if (
        type(sources) is not tuple
        or labels.shape != (len(sources),)
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or any(source not in SOURCES for source in sources)
        or type(base_results) is not dict
        or type(candidate) is not EnsembleCandidateV9
        or candidate not in ensemble_candidate_registry_v9()
    ):
        raise ValueError("shared ensemble inputs differ from the frozen contract")
    required = tuple(
        (candidate_id, seed)
        for candidate_id in candidate.member_candidate_ids
        for seed in candidate.seeds
    )
    if any(
        key not in base_results or type(base_results[key]) is not ResidualEvaluationV8
        for key in required
    ):
        raise ValueError("shared ensemble is missing a frozen base evaluation")
    rows = np.stack([base_results[key].probabilities for key in required])
    if (
        rows.shape != (len(required), len(labels))
        or not np.isfinite(rows).all()
        or np.any((rows <= 0.0) | (rows >= 1.0))
        or any(
            base_results[key].shared_gradient_sources != SOURCES for key in required
        )
    ):
        raise ValueError("base probabilities or shared gradients are incomplete")
    if candidate.aggregation == "probability_mean":
        probabilities = rows.mean(axis=0)
    elif candidate.aggregation == "logit_mean":
        logits = np.log(rows) - np.log1p(-rows)
        mean_logits = logits.mean(axis=0)
        probabilities = 1.0 / (1.0 + np.exp(-mean_logits))
    else:
        raise AssertionError("unreachable ensemble aggregation")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in sources])
        metrics[source] = binary_metrics(labels[selected], probabilities[selected])
    return EnsembleEvaluationV9(
        probabilities=_immutable(probabilities.astype(np.float64)),
        metrics=metrics,
        member_models=len(required),
        model_fits=sum(base_results[key].model_fits for key in required),
        shared_gradient_sources=SOURCES,
    )


def rank_ensemble_results(
    results: dict[str, EnsembleEvaluationV9],
) -> tuple[str, ...]:
    expected = {row.candidate_id for row in ensemble_candidate_registry_v9()}
    if (
        type(results) is not dict
        or set(results) != expected
        or any(type(value) is not EnsembleEvaluationV9 for value in results.values())
    ):
        raise ValueError("ensemble ranking requires the full frozen registry")
    comparator = results["SEN9-000"]

    def key(candidate_id: str):
        metrics = results[candidate_id].metrics
        feasible = all(
            metrics[source]["sensitivity"] + 1e-12 >= 0.85
            and metrics[source]["accuracy"] + 0.01 + 1e-12
            >= comparator.metrics[source]["accuracy"]
            and metrics[source]["auroc"] + 0.01 + 1e-12
            >= comparator.metrics[source]["auroc"]
            for source in SOURCES
        )
        specificity = [metrics[source]["specificity"] for source in SOURCES]
        auroc = [metrics[source]["auroc"] for source in SOURCES]
        accuracy = [metrics[source]["accuracy"] for source in SOURCES]
        balanced = [metrics[source]["balanced_accuracy"] for source in SOURCES]
        return (
            not feasible,
            -min(specificity),
            -min(auroc),
            -min(accuracy),
            -min(balanced),
            -float(np.mean(accuracy)),
            candidate_id,
        )

    return tuple(sorted(expected, key=key))


__all__ = [
    "EnsembleCandidateV9",
    "EnsembleEvaluationV9",
    "ensemble_candidate_registry_v9",
    "evaluate_ensemble_candidate",
    "rank_ensemble_results",
]
