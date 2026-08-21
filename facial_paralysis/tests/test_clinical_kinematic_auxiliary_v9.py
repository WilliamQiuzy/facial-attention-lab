from __future__ import annotations

import inspect

import torch

from _testlib import run_all

from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)
from src.training.clinical_kinematic_auxiliary_v9 import (
    KINEMATIC_TARGET_NAMES,
    ClinicalKinematicAuxiliaryHeadV9,
    clinical_kinematic_auxiliary_loss,
    clinical_kinematic_targets,
    fit_kinematic_target_scaler,
)


def _evidence():
    original = torch.zeros(2, 3, 110, dtype=torch.float32)
    mirrored = torch.zeros_like(original)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    names = candidate_feature_names(LANDMARK_MI_110D)
    for index, name in enumerate(names):
        if name.startswith(("fissure_h_", "fissure_w_", "eye_area_")):
            if name.endswith("__range"):
                original[..., index] = mirrored[..., index] = 2.0
            elif name.endswith("__max_abs_velocity_per_second"):
                original[..., index] = mirrored[..., index] = 3.0
            elif name.endswith("__correlation"):
                original[..., index] = mirrored[..., index] = 0.5
        elif name.startswith("brow_h_"):
            if name.endswith("__range"):
                original[..., index] = mirrored[..., index] = 4.0
            elif name.endswith("__max_abs_velocity_per_second"):
                original[..., index] = mirrored[..., index] = 5.0
            elif name.endswith("__correlation"):
                original[..., index] = mirrored[..., index] = 0.6
        else:
            if name.endswith("__range"):
                original[..., index] = mirrored[..., index] = 6.0
            elif name.endswith("__max_abs_velocity_per_second"):
                original[..., index] = mirrored[..., index] = 7.0
            elif name.endswith("__correlation"):
                original[..., index] = mirrored[..., index] = 0.7
    return original, mirrored, mask, names


def test_targets_are_exact_label_free_clinical_kinematics(c):
    original, mirrored, mask, names = _evidence()
    targets = clinical_kinematic_targets(original, mirrored, mask, names)
    c.eq(tuple(targets.shape), (2, 3, 9))
    c.eq(
        KINEMATIC_TARGET_NAMES,
        (
            "eye_excursion", "eye_velocity", "eye_bilateral_synchrony",
            "brow_excursion", "brow_velocity", "brow_bilateral_synchrony",
            "oral_excursion", "oral_velocity", "oral_bilateral_synchrony",
        ),
    )
    c.true(torch.allclose(
        targets[0, 0],
        torch.tensor([2.0, 3.0, 0.5, 4.0, 5.0, 0.6, 6.0, 7.0, 0.7]),
        atol=1e-6,
    ))
    c.true(bool(torch.equal(targets[~mask], torch.zeros_like(targets[~mask]))))
    c.true("labels" not in inspect.signature(clinical_kinematic_targets).parameters)


def test_targets_are_exactly_invariant_to_original_mirror_order(c):
    original, mirrored, mask, names = _evidence()
    mirrored = mirrored + torch.linspace(0.0, 0.2, 110)
    first = clinical_kinematic_targets(original, mirrored, mask, names)
    second = clinical_kinematic_targets(mirrored, original, mask, names)
    c.true(torch.equal(first, second))


def test_scaler_and_loss_use_only_valid_training_actions(c):
    original, mirrored, mask, names = _evidence()
    targets = clinical_kinematic_targets(original, mirrored, mask, names)
    poisoned = targets.clone()
    poisoned[~mask] = 1e6
    first = fit_kinematic_target_scaler(targets, mask)
    second = fit_kinematic_target_scaler(poisoned, mask)
    c.true(torch.equal(first.mean, second.mean))
    c.true(torch.equal(first.scale, second.scale))
    tokens = torch.randn(2, 3, 64, generator=torch.Generator().manual_seed(7))
    head = ClinicalKinematicAuxiliaryHeadV9()
    loss_a = clinical_kinematic_auxiliary_loss(head, tokens, targets, mask, first)
    loss_b = clinical_kinematic_auxiliary_loss(head, tokens, poisoned, mask, first)
    c.true(torch.equal(loss_a, loss_b))
    c.true(bool(torch.isfinite(loss_a)) and float(loss_a) >= 0.0)


def test_schema_drift_and_empty_support_fail_closed(c):
    original, mirrored, mask, names = _evidence()
    reordered = list(names)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    c.raises(
        lambda: clinical_kinematic_targets(
            original, mirrored, mask, tuple(reordered)
        ),
        ValueError,
    )
    c.raises(
        lambda: fit_kinematic_target_scaler(
            torch.zeros(1, 1, 9), torch.zeros(1, 1, dtype=torch.bool)
        ),
        ValueError,
    )


if __name__ == "__main__":
    run_all("test_clinical_kinematic_auxiliary_v9", dict(globals()))
