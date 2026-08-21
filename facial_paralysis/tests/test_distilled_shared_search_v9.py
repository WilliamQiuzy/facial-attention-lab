from __future__ import annotations

import inspect

import numpy as np

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.distilled_shared_search_v9 import (
    build_cross_fitted_teacher_targets,
    evaluate_distilled_candidate,
    mechanism_feature_matrix,
)
from src.evaluation.medically_gated_shared_search_v2 import MedicalSharedDatasetV2
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.evaluation.shared_clinical_encoder_v1 import SharedClinicalDataset
from src.models.distilled_shared_candidate_registry_v9 import candidate_registry_v9
from src.models.residual_shared_router_v8 import candidate_registry_v8


def _replace_labels(
    dataset: MedicalSharedDatasetV2,
    labels: np.ndarray,
) -> MedicalSharedDatasetV2:
    base = dataset.base
    changed = SharedClinicalDataset(
        clinical_original=base.clinical_original,
        clinical_mirrored=base.clinical_mirrored,
        dense_original=base.dense_original,
        dense_mirrored=base.dense_mirrored,
        dense_valid_mask=base.dense_valid_mask,
        dense_available=base.dense_available,
        action_mask=base.action_mask,
        action_codes=base.action_codes,
        labels=np.asarray(labels, dtype=np.int64),
        group_ids=base.group_ids,
        sources=base.sources,
    )
    return MedicalSharedDatasetV2(changed, dataset.dense_timestamps)


def test_registry_is_bounded_medical_and_keeps_exact_v8_comparator(c):
    registry = candidate_registry_v9()
    c.eq(len(registry), 13)
    c.eq(registry[0].candidate_id, "DSR9-000")
    c.eq(registry[0].teacher_mode, "off")
    c.eq(registry[0].distillation_weight, 0.0)
    c.eq(len({row.candidate_id for row in registry}), len(registry))
    c.true(all(row.medical_rationale for row in registry))
    c.true(all(row.teacher_mode != "off" for row in registry[1:]))


def test_mechanism_features_are_finite_source_blind_and_medically_partitioned(c):
    dataset = _dataset(per_cell=4)
    clinical = mechanism_feature_matrix(dataset, include_dense=False)
    combined = mechanism_feature_matrix(dataset, include_dense=True)
    c.eq(clinical.shape, (24, 2860))
    c.eq(combined.shape, (24, 3068))
    c.true(np.isfinite(clinical).all() and np.isfinite(combined).all())
    c.true(not clinical.flags.writeable and not combined.flags.writeable)
    c.true("sources" not in inspect.signature(mechanism_feature_matrix).parameters)


def test_teacher_targets_are_inner_oof_and_ignore_outer_held_labels(c):
    dataset = _dataset(per_cell=4)
    train = np.asarray([
        0, 1, 2, 4, 5, 6,
        8, 9, 10, 12, 13, 14,
        16, 17, 18, 20, 21, 22,
    ], dtype=np.int64)
    first = build_cross_fitted_teacher_targets(
        dataset, train, teacher_mode="clinical_logistic_32", inner_splits=2,
    )
    labels = dataset.base.labels.copy()
    held = np.setdiff1d(np.arange(24, dtype=np.int64), train)
    labels[held] = 1 - labels[held]
    second = build_cross_fitted_teacher_targets(
        _replace_labels(dataset, labels),
        train,
        teacher_mode="clinical_logistic_32",
        inner_splits=2,
    )
    c.true(np.array_equal(first.probabilities, second.probabilities))
    c.eq(first.training_indices, tuple(int(index) for index in train))
    c.eq(first.self_training_rows, 0)
    c.eq(first.inner_held_group_overlap, 0)
    c.true(np.all((first.probabilities > 0) & (first.probabilities < 1)))


def test_distilled_evaluation_remains_shared_and_participant_disjoint(c):
    candidate = candidate_registry_v9()[1]
    result = evaluate_distilled_candidate(
        _dataset(per_cell=4), candidate, epochs=1, n_splits=2,
        teacher_inner_splits=2, seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (24,))
    c.eq(result.model_fits, 2)
    c.eq(result.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.eq(result.teacher_self_training_rows, 0)
    c.eq(result.outer_held_teacher_reads, 0)
    c.true(result.task_specific_parameter_fraction < 0.10)


def test_off_candidate_is_exact_deterministic_v8_training_comparator(c):
    dataset = _dataset()
    observed = evaluate_distilled_candidate(
        dataset, candidate_registry_v9()[0], epochs=1, n_splits=2,
        teacher_inner_splits=2, seed=0, device="cpu",
    )
    baseline = evaluate_residual_candidate(
        dataset, candidate_registry_v8()[1], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.true(np.array_equal(observed.probabilities, baseline.probabilities))


if __name__ == "__main__":
    run_all("test_distilled_shared_search_v9", dict(globals()))
