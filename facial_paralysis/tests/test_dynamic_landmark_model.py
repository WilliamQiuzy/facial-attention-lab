"""Contract tests for the equal-shape dynamic landmark neural model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES  # noqa: E402
from src.models.dynamic_landmark import (  # noqa: E402
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
    DynamicLandmarkModel,
    gap_safe_per_second_differences,
    horizontal_mirror_features,
)


def _batch(batch_size: int = 2):
    generator = torch.Generator().manual_seed(1701)
    features = torch.randn(batch_size, 4, 32, 95, generator=generator)
    mask = torch.ones(batch_size, 4, 32, dtype=torch.bool)
    indices = torch.arange(128, dtype=torch.int64).reshape(4, 32)
    indices = indices.unsqueeze(0).repeat(batch_size, 1, 1)
    timestamps = indices.to(torch.float32) / 30.0
    return features, mask, timestamps, indices


def _parameter_shapes(model: DynamicLandmarkModel):
    return {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}


def test_all_arms_have_identical_exact_architecture(c: Check):
    models = [DynamicLandmarkModel(arm) for arm in (
        ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION
    )]
    c.true(all(_parameter_shapes(model) == _parameter_shapes(models[0])
               for model in models[1:]),
           "every ablation has identical parameter names and shapes")
    model = models[0]
    c.eq((model.proj_bs_x.in_features, model.proj_bs_x.out_features), (72, 32))
    c.eq((model.proj_bs_dx.in_features, model.proj_bs_dx.out_features), (72, 32))
    c.eq((model.proj_lm_x.in_features, model.proj_lm_x.out_features), (23, 32))
    c.eq((model.proj_lm_dx.in_features, model.proj_lm_dx.out_features), (23, 32))
    c.true(all(layer.bias is None for layer in (
        model.proj_bs_x, model.proj_bs_dx, model.proj_lm_x, model.proj_lm_dx
    )), "all four frame projections are bias-free")
    c.eq(model.temporal.input_size, 64, "concatenated frame latent is exactly 64")
    c.eq(model.temporal.hidden_size, 32, "GRU hidden width is 32 per direction")
    c.true(model.temporal.bidirectional and model.temporal.num_layers == 1,
           "temporal encoder is a one-layer BiGRU")
    c.eq((model.pool_projection.in_features, model.pool_projection.out_features),
         (128, 32), "max and attention pools concatenate before 32-d projection")


def test_inactive_blocks_cannot_affect_output_and_active_blocks_get_gradients(c: Check):
    features, mask, timestamps, indices = _batch()
    for arm, inactive_slice, active_prefixes in (
        (ARM_BLENDSHAPE, slice(72, 95), ("proj_bs_x", "proj_bs_dx")),
        (ARM_LANDMARK, slice(0, 72), ("proj_lm_x", "proj_lm_dx")),
    ):
        torch.manual_seed(33)
        model = DynamicLandmarkModel(arm).eval()
        changed = features.clone()
        changed[..., inactive_slice] = changed[..., inactive_slice] * 1e5 + 999.0
        first = model(features, mask, timestamps, indices)
        second = model(changed, mask, timestamps, indices)
        c.true(bool(torch.equal(first, second)), f"{arm} ignores its inactive block")
        model.zero_grad(set_to_none=True)
        model(features, mask, timestamps, indices).sum().backward()
        grads = dict(model.named_parameters())
        for prefix in active_prefixes:
            gradient = grads[f"{prefix}.weight"].grad
            c.true(gradient is not None and float(gradient.abs().sum()) > 0,
                   f"{arm} active {prefix} receives gradient")
        inactive = ("proj_lm_x", "proj_lm_dx") if arm == ARM_BLENDSHAPE else (
            "proj_bs_x", "proj_bs_dx"
        )
        for prefix in inactive:
            gradient = grads[f"{prefix}.weight"].grad
            c.true(gradient is not None and float(gradient.abs().sum()) == 0.0,
                   f"{arm} inactive {prefix} has exactly zero gradient")


def test_gap_safe_per_second_differences_never_bridge_mask_or_frame_gaps(c: Check):
    values = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0]).reshape(1, 1, 5, 1)
    mask = torch.ones(1, 1, 5, dtype=torch.bool)
    times = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]).reshape(1, 1, 5)
    source = torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64).reshape(1, 1, 5)
    delta, delta_mask = gap_safe_per_second_differences(values, mask, times, source)
    c.true(bool(torch.allclose(delta.flatten(), torch.tensor([0., 10., 0., 30., 40.]))),
           "derivatives are per second and source gaps stay zero")
    c.eq(delta_mask.flatten().tolist(), [False, True, False, True, True])

    mask[..., 3] = False
    delta, delta_mask = gap_safe_per_second_differences(values, mask, times, source)
    c.eq(delta_mask.flatten().tolist(), [False, True, False, False, False],
         "a detector gap invalidates both adjacent differences")
    c.true(bool((delta[~delta_mask] == 0).all()), "invalid differences are canonical zero")

    repeated = source.clone()
    repeated[..., 3] = repeated[..., 2]
    c.raises(lambda: gap_safe_per_second_differences(values, mask, times, repeated),
             ValueError, "repeated source indices fail closed")
    decreasing = source.clone()
    decreasing[..., 3] = decreasing[..., 2] - 1
    c.raises(lambda: gap_safe_per_second_differences(values, mask, times, decreasing),
             ValueError, "decreasing source indices fail closed")


def test_horizontal_mirror_uses_exact_schema_swaps_and_signs(c: Check):
    vector = torch.arange(95, dtype=torch.float32)
    mirrored = horizontal_mirror_features(vector)
    names = list(DYNAMIC_FEATURE_NAMES)
    by_name = {name: mirrored[index].item() for index, name in enumerate(names)}
    original = {name: vector[index].item() for index, name in enumerate(names)}
    c.eq(by_name["eyeBlinkLeft"], original["eyeBlinkRight"])
    c.eq(by_name["jawLeft"], original["jawRight"])
    c.eq(by_name["delta_left_minus_right_eyeBlink"],
         -original["delta_left_minus_right_eyeBlink"])
    c.eq(by_name["fissure_h_mesh33"], original["fissure_h_mesh263"])
    c.eq(by_name["corner_x_mesh61"], original["corner_x_mesh291"])
    for signed in (
        "fissure_h_mesh33_minus_mesh263", "brow_h_mesh33_minus_mesh263",
        "corner_y_mesh61_minus_mesh291",
    ):
        c.eq(by_name[signed], -original[signed], f"{signed} changes sign")
    for invariant in ("_neutral", "fissure_h_absdiff", "mouth_width", "mouth_open"):
        c.eq(by_name[invariant], original[invariant], f"{invariant} is invariant")
    c.true(bool(torch.equal(horizontal_mirror_features(mirrored), vector)),
           "horizontal mirror is an exact involution")
    c.true(bool(torch.equal(vector, torch.arange(95, dtype=torch.float32))),
           "augmentation never mutates its input")


def test_forward_masks_absent_windows_rejects_bad_inputs_and_is_deterministic(c: Check):
    features, mask, timestamps, indices = _batch(batch_size=1)
    mask[:, 3] = False
    features[:, 3] = 12345.0
    torch.manual_seed(91)
    model = DynamicLandmarkModel(ARM_FUSION).eval()
    first = model(features, mask, timestamps, indices)
    changed = features.clone()
    changed[:, 3] = -98765.0
    second = model(changed, mask, timestamps, indices)
    third = model(features, mask, timestamps, indices)
    c.eq(tuple(first.shape), (1,), "one binary logit per recording")
    c.true(bool(torch.equal(first, second)), "absent windows do not enter recording mean")
    c.true(bool(torch.equal(first, third)), "evaluation inference is deterministic")

    c.raises(lambda: model(features, torch.zeros_like(mask), timestamps, indices),
             ValueError, "an all-masked recording is rejected")
    nonfinite = features.clone()
    nonfinite[0, 3, 0, 0] = float("nan")
    c.raises(lambda: model(nonfinite, mask, timestamps, indices), ValueError,
             "nonfinite values are rejected even in masked rows")
    c.raises(lambda: horizontal_mirror_features(torch.zeros(94)), ValueError,
             "mirror augmentation requires the exact schema width")

    foreign_mask = torch.empty(mask.shape, dtype=torch.bool, device="meta")
    c.raises(lambda: model(features, foreign_mask, timestamps, indices), ValueError,
             "all four input tensors must share one device")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_model", dict(globals()))
