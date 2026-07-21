"""Protocol-contract tests for the focused Fusion robustness benchmark."""
from __future__ import annotations

import inspect
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.focused_fusion_robustness import (  # noqa: E402
    BENCHMARK_CONDITIONS,
    BenchmarkCondition,
    build_condition_inputs,
)


class _HostileEqual:
    def __init__(self):
        self.comparisons = 0

    def __eq__(self, _other):
        self.comparisons += 1
        return True


def _condition(name: str) -> BenchmarkCondition:
    return next(item for item in BENCHMARK_CONDITIONS if item.name == name)


def _synthetic_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.arange(
        2 * 4 * 32 * 95, dtype=torch.float32,
    ).reshape(2, 4, 32, 95).div(100.0)
    valid_mask = torch.zeros((2, 4, 32), dtype=torch.bool)
    target_mask = torch.zeros_like(valid_mask)

    valid_mask[0, 0, :6] = True
    target_mask[0, 0, (1, 4)] = True
    valid_mask[0, 1, :4] = True
    target_mask[0, 1, 2] = True
    valid_mask[0, 2, 0] = True

    valid_mask[1, 0, :7] = True
    target_mask[1, 0, (0, 5)] = True
    valid_mask[1, 2, :5] = True
    target_mask[1, 2, 3] = True
    valid_mask[1, 3, :3] = True
    target_mask[1, 3, 1] = True
    return features, valid_mask, target_mask


def test_registry_has_exact_condition_order(c: Check):
    c.true(isinstance(BENCHMARK_CONDITIONS, tuple), "registry is immutable")
    c.eq(
        tuple(condition.name for condition in BENCHMARK_CONDITIONS),
        (
            "clean_fusion",
            "mask_landmarks",
            "mask_blendshapes",
            "context_dropout_10pct",
            "context_dropout_25pct",
            "context_dropout_50pct",
            "landmark_noise_0.10sd",
            "landmark_noise_0.25sd",
            "landmark_noise_0.50sd",
            "frame_order_shuffle",
        ),
        "condition names and order are frozen",
    )


def test_registry_has_exact_modality_arms(c: Check):
    c.eq(
        tuple(condition.input_arm for condition in BENCHMARK_CONDITIONS),
        (
            "fusion",
            "blendshape_only",
            "landmark_only",
            "fusion",
            "fusion",
            "fusion",
            "fusion",
            "fusion",
            "fusion",
            "fusion",
        ),
        "modality removal reuses the existing model input arms",
    )


def test_registry_has_exact_probabilities_noise_levels_and_seeds(c: Check):
    c.eq(
        tuple(
            (
                condition.name,
                condition.context_dropout_probability,
                condition.landmark_noise_sd,
                condition.rng_seed,
            )
            for condition in BENCHMARK_CONDITIONS
        ),
        (
            ("clean_fusion", None, None, None),
            ("mask_landmarks", None, None, None),
            ("mask_blendshapes", None, None, None),
            ("context_dropout_10pct", 0.10, None, 41010),
            ("context_dropout_25pct", 0.25, None, 41025),
            ("context_dropout_50pct", 0.50, None, 41050),
            ("landmark_noise_0.10sd", None, 0.10, 52010),
            ("landmark_noise_0.25sd", None, 0.25, 52025),
            ("landmark_noise_0.50sd", None, 0.50, 52050),
            ("frame_order_shuffle", None, None, 63000),
        ),
        "all optional perturbation parameters are exact and explicit",
    )


def test_conditions_are_frozen(c: Check):
    clean = BENCHMARK_CONDITIONS[0]
    c.true(all(isinstance(item, BenchmarkCondition)
               for item in BENCHMARK_CONDITIONS),
           "registry entries use the frozen protocol type")
    c.raises(lambda: setattr(clean, "name", "caller_defined"),
             FrozenInstanceError, "registered conditions cannot be mutated")


def test_conditions_have_no_mutable_instance_dictionary(c: Check):
    clean = BENCHMARK_CONDITIONS[0]
    c.raises(lambda: clean.__dict__, AttributeError,
             "indirect instance-dictionary mutation is unavailable")


def test_condition_type_cannot_be_subclassed(c: Check):
    def construct_bypass():
        subclass = type(
            "BypassBenchmarkCondition",
            (BenchmarkCondition,),
            {"__post_init__": lambda _self: None},
        )
        return subclass(name="caller_defined", input_arm="fusion")

    c.raises(construct_bypass, TypeError,
             "subclasses cannot override validation")


def test_hostile_equality_is_rejected_before_comparison(c: Check):
    for field in (
        "name",
        "input_arm",
        "context_dropout_probability",
        "landmark_noise_sd",
        "rng_seed",
    ):
        hostile = _HostileEqual()
        kwargs = {"name": "clean_fusion", "input_arm": "fusion"}
        kwargs[field] = hostile
        c.raises(lambda kwargs=kwargs: BenchmarkCondition(**kwargs), TypeError,
                 f"{field} requires an exact scalar type")
        c.eq(hostile.comparisons, 0,
             f"{field} is type-checked before equality")


def test_wrong_scalar_types_fail_closed(c: Check):
    invalid_specs = (
        {"name": b"clean_fusion", "input_arm": "fusion"},
        {"name": "clean_fusion", "input_arm": b"fusion"},
        {
            "name": "context_dropout_10pct",
            "input_arm": "fusion",
            "context_dropout_probability": 1,
            "rng_seed": 41010,
        },
        {
            "name": "landmark_noise_0.10sd",
            "input_arm": "fusion",
            "landmark_noise_sd": "0.10",
            "rng_seed": 52010,
        },
        {
            "name": "landmark_noise_0.10sd",
            "input_arm": "fusion",
            "landmark_noise_sd": 0.10,
            "rng_seed": 52010.0,
        },
        {
            "name": "frame_order_shuffle",
            "input_arm": "fusion",
            "rng_seed": True,
        },
    )
    for kwargs in invalid_specs:
        c.raises(lambda kwargs=kwargs: BenchmarkCondition(**kwargs), TypeError,
                 "wrong scalar types, including bool-as-int, fail closed")


def test_caller_defined_protocols_fail(c: Check):
    clean = BENCHMARK_CONDITIONS[0]
    c.raises(lambda: BenchmarkCondition(
        name="caller_defined",
        input_arm="fusion",
    ), ValueError, "callers cannot add arbitrary condition names")
    c.raises(lambda: replace(clean, context_dropout_probability=0.25),
             ValueError, "callers cannot alter a registered condition spec")


def test_condition_input_api_keeps_temporal_metadata_outside(c: Check):
    c.eq(
        tuple(inspect.signature(build_condition_inputs).parameters),
        ("features", "valid_mask", "target_mask", "condition"),
        "timestamps and source indices remain outside the perturbation API",
    )


def test_clean_and_modality_conditions_change_only_the_input_arm(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    expected = (
        ("clean_fusion", "fusion"),
        ("mask_landmarks", "blendshape_only"),
        ("mask_blendshapes", "landmark_only"),
    )
    for name, expected_arm in expected:
        model_features, reconstruction_mask, input_arm = build_condition_inputs(
            features, valid_mask, target_mask, _condition(name),
        )
        c.true(torch.equal(model_features, features),
               f"{name} does not zero or otherwise edit feature storage")
        c.true(torch.equal(reconstruction_mask, target_mask),
               f"{name} keeps the clean target mask")
        c.eq(input_arm, expected_arm,
             f"{name} expresses modality selection through the model arm")


def test_every_condition_is_deterministic_context_only_and_non_mutating(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    clean_features = features.clone()
    clean_valid_mask = valid_mask.clone()
    clean_target_mask = target_mask.clone()
    observed_context = valid_mask & ~target_mask

    for condition in BENCHMARK_CONDITIONS:
        first = build_condition_inputs(
            features, valid_mask, target_mask, condition,
        )
        second = build_condition_inputs(
            features, valid_mask, target_mask, condition,
        )
        first_features, first_reconstruction, first_arm = first
        second_features, second_reconstruction, second_arm = second
        c.true(torch.equal(first_features, second_features),
               f"{condition.name} feature bytes are deterministic")
        c.true(torch.equal(first_reconstruction, second_reconstruction),
               f"{condition.name} reconstruction mask is deterministic")
        c.eq(first_arm, second_arm,
             f"{condition.name} input arm is deterministic")

        changed_positions = (first_features != features).any(dim=-1)
        c.true(not bool((changed_positions & ~observed_context).any()),
               f"{condition.name} edits only observed context")
        c.true(torch.equal(first_features[target_mask], features[target_mask]),
               f"{condition.name} leaves clean target values exact")
        c.true(torch.equal(first_features[~valid_mask], features[~valid_mask]),
               f"{condition.name} preserves invalid feature storage")
        added_reconstruction = first_reconstruction & ~target_mask
        c.true(not bool((added_reconstruction & ~observed_context).any()),
               f"{condition.name} masks only observed context")
        remaining = valid_mask & ~first_reconstruction
        c.true(bool(remaining.reshape(features.shape[0], -1).any(dim=1).all()),
               f"{condition.name} retains context for every sample")

        c.true(torch.equal(features, clean_features),
               f"{condition.name} does not mutate caller feature storage")
        c.true(torch.equal(valid_mask, clean_valid_mask),
               f"{condition.name} does not mutate the valid mask")
        c.true(torch.equal(target_mask, clean_target_mask),
               f"{condition.name} does not mutate the clean target mask")


def test_context_dropout_uses_full_shape_cpu_draw_and_model_only_mask(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    observed_context = valid_mask & ~target_mask
    for condition in BENCHMARK_CONDITIONS[3:6]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(condition.rng_seed)
        random_draw = torch.rand(
            valid_mask.shape, dtype=torch.float32, generator=generator,
        )
        expected_drop = observed_context & (
            random_draw < condition.context_dropout_probability
        )
        model_features, reconstruction_mask, input_arm = build_condition_inputs(
            features, valid_mask, target_mask, condition,
        )
        c.true(torch.equal(model_features, features),
               "dropout changes only model masking, never clean features")
        c.true(torch.equal(reconstruction_mask, target_mask | expected_drop),
               f"{condition.name} uses its exact full-shape random draw")
        c.eq(input_arm, "fusion", "dropout retains the Fusion arm")


def test_context_dropout_fails_if_a_sample_loses_all_context(c: Check):
    features = torch.arange(
        4 * 32 * 95, dtype=torch.float32,
    ).reshape(1, 4, 32, 95)
    valid_mask = torch.zeros((1, 4, 32), dtype=torch.bool)
    target_mask = torch.zeros_like(valid_mask)
    condition = _condition("context_dropout_50pct")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(condition.rng_seed)
    draw = torch.rand(valid_mask.shape, generator=generator)
    context_flat = int((draw.reshape(-1) < 0.50).nonzero()[0, 0])
    target_flat = (context_flat + 1) % draw.numel()
    valid_mask.reshape(-1)[context_flat] = True
    valid_mask.reshape(-1)[target_flat] = True
    target_mask.reshape(-1)[target_flat] = True
    clean_features = features.clone()
    clean_target_mask = target_mask.clone()

    c.raises(
        lambda: build_condition_inputs(
            features, valid_mask, target_mask, condition,
        ),
        ValueError,
        "dropout rejects a model mask with no context for one sample",
    )
    c.true(torch.equal(features, clean_features),
           "failed dropout leaves caller feature storage unchanged")
    c.true(torch.equal(target_mask, clean_target_mask),
           "failed dropout leaves the clean target mask unchanged")


def test_landmark_noise_uses_full_block_draw_and_only_observed_landmarks(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    observed_context = valid_mask & ~target_mask
    for condition in BENCHMARK_CONDITIONS[6:9]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(condition.rng_seed)
        noise = torch.randn(
            features[..., 72:95].shape,
            dtype=features.dtype,
            generator=generator,
        ).mul(condition.landmark_noise_sd)
        expected = features.clone()
        expected_landmarks = expected[..., 72:95]
        expected_landmarks[observed_context] += noise[observed_context]

        model_features, reconstruction_mask, input_arm = build_condition_inputs(
            features, valid_mask, target_mask, condition,
        )
        c.true(torch.equal(model_features, expected),
               f"{condition.name} uses the exact full landmark-block draw")
        c.true(torch.equal(model_features[..., :72], features[..., :72]),
               f"{condition.name} preserves all BS72 storage")
        c.true(torch.equal(reconstruction_mask, target_mask),
               f"{condition.name} retains the clean reconstruction mask")
        c.eq(input_arm, "fusion", "noise retains the Fusion arm")


def test_frame_shuffle_is_sample_major_window_major_and_row_complete(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    observed_context = valid_mask & ~target_mask
    expected = features.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(63000)
    for sample in range(features.shape[0]):
        for window in range(features.shape[1]):
            context_indices = observed_context[sample, window].nonzero().flatten()
            permutation = torch.randperm(
                context_indices.numel(), generator=generator,
            )
            expected[sample, window, context_indices] = features[
                sample, window, context_indices[permutation]
            ]

    model_features, reconstruction_mask, input_arm = build_condition_inputs(
        features, valid_mask, target_mask, _condition("frame_order_shuffle"),
    )
    c.true(torch.equal(model_features, expected),
           "shuffle advances sample-major then window-major over complete rows")
    c.true(torch.equal(reconstruction_mask, target_mask),
           "shuffle retains the clean reconstruction mask")
    c.eq(input_arm, "fusion", "shuffle retains the Fusion arm")


def test_condition_inputs_fail_closed_on_invalid_tensor_contracts(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    clean = _condition("clean_fusion")
    invalid_calls = (
        lambda: build_condition_inputs(
            features[0], valid_mask[0], target_mask[0], clean,
        ),
        lambda: build_condition_inputs(
            features.to(torch.int64), valid_mask, target_mask, clean,
        ),
        lambda: build_condition_inputs(
            features.clone().index_fill(-1, torch.tensor([0]), float("nan")),
            valid_mask,
            target_mask,
            clean,
        ),
        lambda: build_condition_inputs(
            features, valid_mask.to(torch.float32), target_mask, clean,
        ),
        lambda: build_condition_inputs(
            features, valid_mask, target_mask[..., :-1], clean,
        ),
        lambda: build_condition_inputs(
            features,
            valid_mask,
            target_mask | ~valid_mask,
            clean,
        ),
        lambda: build_condition_inputs(
            features,
            valid_mask,
            valid_mask.clone(),
            clean,
        ),
        lambda: build_condition_inputs(
            features,
            valid_mask,
            target_mask,
            None,
        ),
    )
    for invalid_call in invalid_calls:
        c.raises(invalid_call, ValueError,
                 "invalid tensors and non-condition objects fail closed")


def test_exact_registered_spec_instances_are_accepted(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    exact_clean_spec = BenchmarkCondition(
        name="clean_fusion", input_arm="fusion",
    )
    model_features, reconstruction_mask, input_arm = build_condition_inputs(
        features, valid_mask, target_mask, exact_clean_spec,
    )
    c.true(torch.equal(model_features, features),
           "an independently constructed exact registered spec is unchanged")
    c.true(torch.equal(reconstruction_mask, target_mask),
           "an exact registered spec keeps the target mask")
    c.eq(input_arm, "fusion", "an exact registered spec retains its arm")


if __name__ == "__main__":
    run_all("test_focused_fusion_robustness", dict(globals()))
