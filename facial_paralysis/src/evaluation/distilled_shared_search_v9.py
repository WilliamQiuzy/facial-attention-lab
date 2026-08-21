"""Nested, participant-disjoint privileged distillation for a shared V9 encoder."""
from __future__ import annotations

from dataclasses import dataclass
import os
import warnings

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
from src.models.distilled_shared_candidate_registry_v9 import (
    DistilledSharedCandidateV9,
    candidate_registry_v9,
)
from src.models.medically_gated_shared_encoder_v2 import (
    BROW_LANDMARKS,
    EYE_LANDMARKS,
    MOUTH_LANDMARKS,
)
from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_ACTION_COUNT = 13
_CLINICAL_PER_ACTION = 220
_DENSE_PER_ACTION = 16
_REGIONS = (
    BROW_LANDMARKS,
    EYE_LANDMARKS,
    MOUTH_LANDMARKS,
    tuple(range(478)),
)
_TEACHER_MODES = {
    "clinical_logistic_32": (False, "logistic", 32),
    "clinical_logistic_64": (False, "logistic", 64),
    "mechanism_logistic_64": (True, "logistic", 64),
    "mechanism_rbf_64": (True, "rbf", 64),
}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def configure_deterministic_training_v9(device: torch.device) -> None:
    if type(device) is not torch.device or device.type not in {"cpu", "cuda"}:
        raise ValueError("deterministic V9 requires one CPU or CUDA device")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA V9 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _clinical_features(dataset: MedicalSharedDatasetV2) -> np.ndarray:
    base = dataset.base
    output = np.zeros(
        (len(base.labels), _ACTION_COUNT, _CLINICAL_PER_ACTION), dtype=np.float64
    )
    for participant in range(len(base.labels)):
        for action_code in range(_ACTION_COUNT):
            selected = base.action_mask[participant] & (
                base.action_codes[participant] == action_code
            )
            if not selected.any():
                continue
            first = base.clinical_original[participant, selected].astype(np.float64)
            second = base.clinical_mirrored[participant, selected].astype(np.float64)
            output[participant, action_code] = np.concatenate((
                0.5 * (first + second).mean(axis=0),
                np.abs(first - second).mean(axis=0),
            ))
    return output.reshape(len(base.labels), -1)


def _regional_summary(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    selected = trajectory[:, indices, :].astype(np.float64, copy=False)
    relative = selected - selected[:1]
    excursion = np.linalg.norm(relative, axis=-1)
    intervals = np.diff(timestamps).clip(min=1e-6)
    velocity = np.linalg.norm(np.diff(selected, axis=0), axis=-1) / intervals[:, None]
    return np.asarray((
        float(excursion.mean()),
        float(excursion.max()),
        float(velocity.mean()),
        float(np.quantile(velocity, 0.90)),
    ), dtype=np.float64)


def _dense_features(dataset: MedicalSharedDatasetV2) -> np.ndarray:
    base = dataset.base
    output = np.zeros(
        (len(base.labels), _ACTION_COUNT, _DENSE_PER_ACTION), dtype=np.float64
    )
    for participant in range(len(base.labels)):
        for action_code in range(_ACTION_COUNT):
            selected = np.flatnonzero(
                base.action_mask[participant]
                & base.dense_available[participant]
                & (base.action_codes[participant] == action_code)
            )
            if selected.size == 0:
                continue
            repeated = []
            for action in selected.tolist():
                blocks = []
                for region in _REGIONS:
                    first = _regional_summary(
                        base.dense_original[participant, action],
                        dataset.dense_timestamps[participant, action],
                        region,
                    )
                    second = _regional_summary(
                        base.dense_mirrored[participant, action],
                        dataset.dense_timestamps[participant, action],
                        region,
                    )
                    blocks.extend((
                        0.5 * (first[0] + second[0]),
                        0.5 * (first[1] + second[1]),
                        0.5 * (first[2] + second[2]),
                        abs(first[0] - second[0]),
                    ))
                repeated.append(np.asarray(blocks, dtype=np.float64))
            output[participant, action_code] = np.mean(repeated, axis=0)
    return output.reshape(len(base.labels), -1)


def mechanism_feature_matrix(
    dataset: MedicalSharedDatasetV2,
    *,
    include_dense: bool,
) -> np.ndarray:
    if type(dataset) is not MedicalSharedDatasetV2 or type(include_dense) is not bool:
        raise ValueError("mechanism features require exact validated inputs")
    clinical = _clinical_features(dataset)
    result = (
        np.concatenate((clinical, _dense_features(dataset)), axis=1)
        if include_dense else clinical
    )
    if not np.isfinite(result).all():
        raise RuntimeError("mechanism features contain nonfinite values")
    return _immutable(result)


@dataclass(frozen=True)
class TeacherTargetsV9:
    probabilities: np.ndarray
    training_indices: tuple[int, ...]
    self_training_rows: int
    inner_held_group_overlap: int


def _rank_features(matrix: np.ndarray, labels: np.ndarray, top_k: int) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        scores, _ = f_classif(matrix, labels)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return np.lexsort((np.arange(scores.size), -scores))[:top_k]


def _teacher_estimator(kind: str):
    if kind == "logistic":
        model = LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=20260821,
        )
    elif kind == "rbf":
        model = SVC(
            C=1.0, gamma="scale", probability=True, class_weight="balanced",
            random_state=20260821,
        )
    else:
        raise ValueError("unknown teacher estimator")
    return make_pipeline(StandardScaler(), model)


def build_cross_fitted_teacher_targets(
    dataset: MedicalSharedDatasetV2,
    training_indices: np.ndarray,
    *,
    teacher_mode: str,
    inner_splits: int,
) -> TeacherTargetsV9:
    indices = np.asarray(training_indices)
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(training_indices) is not np.ndarray
        or indices.dtype != np.dtype(np.int64)
        or indices.ndim != 1
        or indices.size < 12
        or len(np.unique(indices)) != indices.size
        or np.any(indices < 0)
        or np.any(indices >= len(dataset.base.labels))
        or teacher_mode not in _TEACHER_MODES
        or isinstance(inner_splits, bool)
        or not isinstance(inner_splits, int)
        or inner_splits < 2
    ):
        raise ValueError("invalid cross-fitted teacher configuration")
    include_dense, estimator_kind, top_k = _TEACHER_MODES[teacher_mode]
    features = mechanism_feature_matrix(dataset, include_dense=include_dense)
    probabilities = np.full(indices.size, np.nan, dtype=np.float64)
    local_by_global = {int(value): position for position, value in enumerate(indices)}
    overlap = 0
    base = dataset.base
    for source in SOURCES:
        source_global = np.asarray([
            int(index) for index in indices if base.sources[int(index)] == source
        ], dtype=np.int64)
        labels = base.labels[source_global]
        if (
            source_global.size < 2 * inner_splits
            or min(np.bincount(labels, minlength=2)) < inner_splits
        ):
            raise ValueError("each source class must cover every teacher inner fold")
        splitter = StratifiedKFold(
            inner_splits, shuffle=True, random_state=20260821
        )
        for train_local, held_local in splitter.split(
            np.zeros(source_global.size), labels
        ):
            fit_global = source_global[train_local]
            held_global = source_global[held_local]
            overlap += len(
                {base.group_ids[index] for index in fit_global}
                & {base.group_ids[index] for index in held_global}
            )
            selected = _rank_features(
                features[fit_global], base.labels[fit_global],
                min(top_k, features.shape[1]),
            )
            estimator = _teacher_estimator(estimator_kind)
            estimator.fit(features[fit_global][:, selected], base.labels[fit_global])
            values = estimator.predict_proba(features[held_global][:, selected])[:, 1]
            for global_index, value in zip(held_global, values):
                probabilities[local_by_global[int(global_index)]] = float(value)
    if not np.isfinite(probabilities).all() or overlap != 0:
        raise RuntimeError("teacher OOF reconstruction is incomplete or overlapping")
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return TeacherTargetsV9(
        probabilities=_immutable(probabilities),
        training_indices=tuple(int(index) for index in indices),
        self_training_rows=0,
        inner_held_group_overlap=0,
    )


@dataclass(frozen=True)
class DistilledEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    teacher_self_training_rows: int
    outer_held_teacher_reads: int
    task_specific_parameter_fraction: float


def _locked_v8_candidate():
    return next(
        row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"
    )


def _task_specific_fraction(model: ResidualSharedRouterV8) -> float:
    task_specific = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("adapters")
        or name.startswith("base.task_queries")
        or name.startswith("base.backbone.task_heads")
    )
    return task_specific / sum(parameter.numel() for parameter in model.parameters())


def evaluate_distilled_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: DistilledSharedCandidateV9,
    *,
    epochs: int,
    n_splits: int = 6,
    teacher_inner_splits: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> DistilledEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not DistilledSharedCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid distilled shared evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested V9 runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    task_fraction = np.nan
    teacher_self_training_rows = 0

    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = ResidualSharedRouterV8(_locked_v8_candidate()).to(runtime)
        task_fraction = _task_specific_fraction(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        train_sources = tuple(base.sources[index] for index in train)
        weights = _tensor(
            source_class_balanced_weights(base.labels[train], train_sources).astype(
                np.float32
            ),
            runtime,
        )
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        train_inputs = _model_inputs(dataset, original, mirrored, train, runtime)
        if candidate.teacher_mode == "off":
            teacher = labels
        else:
            teacher_result = build_cross_fitted_teacher_targets(
                dataset, train, teacher_mode=candidate.teacher_mode,
                inner_splits=teacher_inner_splits,
            )
            teacher_self_training_rows += teacher_result.self_training_rows
            teacher = _tensor(
                teacher_result.probabilities.astype(np.float32), runtime
            )

        if fold_index == 0:
            for source in SOURCES:
                local = torch.tensor([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=torch.long, device=runtime)
                model.zero_grad(set_to_none=True)
                local_inputs = tuple(value.index_select(0, local) for value in train_inputs)
                tokens = model.shared_action_tokens(*local_inputs)
                loss = F.binary_cross_entropy_with_logits(
                    model.routed_logits(
                        tokens, local_inputs[-2], tasks.index_select(0, local)
                    ),
                    labels.index_select(0, local),
                )
                loss.backward()
                clinical_gradient = model.base.backbone.clinical_encoder[0].weight.grad
                patient_gradient = model.base.backbone.patient_projection.weight.grad
                if (
                    clinical_gradient is None
                    or patient_gradient is None
                    or float(clinical_gradient.norm()) <= 0.0
                    or float(patient_gradient.norm()) <= 0.0
                ):
                    raise RuntimeError("a source failed the shared-gradient audit")
                covered.add(source)

        alpha = candidate.distillation_weight
        target = (1.0 - alpha) * labels + alpha * teacher
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tokens = model.shared_action_tokens(*train_inputs)
            common = model.base.endpoint_embedding(tokens, train_inputs[-2], tasks)
            endpoint = model.adapt_endpoint(common, tasks)
            universal = model.base.universal_embedding(tokens, train_inputs[-2])
            task_logits = model.base.task_logits_from_embedding(endpoint, tasks)
            universal_logits = model.base.universal_head(universal).squeeze(-1)
            routed = 0.75 * task_logits + 0.25 * universal_logits
            losses = F.binary_cross_entropy_with_logits(
                routed, target, reduction="none"
            ) + 0.5 * F.binary_cross_entropy_with_logits(
                universal_logits, labels, reduction="none"
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
            probabilities[held] = torch.sigmoid(
                model.routed_logits(tokens, held_inputs[-2], held_tasks)
            ).cpu().numpy()
        del model, optimizer

    if (
        not np.isfinite(probabilities).all()
        or covered != set(SOURCES)
        or teacher_self_training_rows != 0
        or not np.isfinite(task_fraction)
        or task_fraction >= 0.10
    ):
        raise RuntimeError("distilled V9 evaluation failed its sharing boundary")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in base.sources])
        metrics[source] = binary_metrics(base.labels[selected], probabilities[selected])
    return DistilledEvaluationV9(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        model_fits=len(folds),
        shared_gradient_sources=SOURCES,
        teacher_self_training_rows=0,
        outer_held_teacher_reads=0,
        task_specific_parameter_fraction=float(task_fraction),
    )


def rank_distilled_results(
    results: dict[str, DistilledEvaluationV9],
) -> tuple[str, ...]:
    expected = {row.candidate_id for row in candidate_registry_v9()}
    if type(results) is not dict or set(results) != expected:
        raise ValueError("distilled ranking requires the complete frozen registry")
    comparator = results["DSR9-000"].metrics

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
    "DistilledEvaluationV9",
    "TeacherTargetsV9",
    "build_cross_fitted_teacher_targets",
    "configure_deterministic_training_v9",
    "evaluate_distilled_candidate",
    "mechanism_feature_matrix",
    "rank_distilled_results",
]
