"""Protocol-contract tests for the focused Fusion robustness benchmark."""
from __future__ import annotations

import inspect
import json
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, Inexact, Rounded, getcontext, setcontext
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.focused_fusion_robustness import (  # noqa: E402
    BENCHMARK_CONDITIONS,
    BenchmarkCondition,
    aggregate_condition_metrics,
    build_condition_inputs,
    canonical_metric,
    evaluate_fusion_conditions,
    require_clean_replay,
    validate_deidentified_payload,
    validate_metric_bundle,
)
from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402


class _HostileEqual:
    def __init__(self):
        self.comparisons = 0

    def __eq__(self, _other):
        self.comparisons += 1
        return True


class _StrSubclass(str):
    pass


class _HostileKey:
    def __init__(self, target: object):
        self.target = target
        self.comparisons = 0
        self.hashes = 0

    def __hash__(self):
        self.hashes += 1
        return hash(self.target)

    def __eq__(self, other):
        self.comparisons += 1
        return other == self.target


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


def test_condition_inputs_require_exact_float32_features(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    rejected_dtypes = [torch.float16, torch.bfloat16, torch.float64]
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    ):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            rejected_dtypes.append(dtype)

    for dtype in rejected_dtypes:
        wrong_dtype = torch.zeros(features.shape, dtype=dtype)
        c.raises(
            lambda wrong_dtype=wrong_dtype: build_condition_inputs(
                wrong_dtype,
                valid_mask,
                target_mask,
                _condition("clean_fusion"),
            ),
            ValueError,
            f"{dtype} is rejected through the controlled validation path",
        )


def test_condition_inputs_reject_an_empty_target_mask(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    target_mask.zero_()
    c.raises(
        lambda: build_condition_inputs(
            features,
            valid_mask,
            target_mask,
            _condition("clean_fusion"),
        ),
        ValueError,
        "the fixed benchmark rejects a globally empty target mask",
    )


def test_condition_inputs_require_a_target_for_every_sample(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    target_mask[1].zero_()
    c.true(bool(target_mask[0].any()),
           "the fixture retains a target in the other sample")
    c.raises(
        lambda: build_condition_inputs(
            features,
            valid_mask,
            target_mask,
            _condition("clean_fusion"),
        ),
        ValueError,
        "every fixed benchmark packet requires a scored target",
    )


def test_condition_outputs_have_independent_caller_storage(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    for condition in BENCHMARK_CONDITIONS:
        model_features, reconstruction_mask, _input_arm = build_condition_inputs(
            features, valid_mask, target_mask, condition,
        )
        c.true(
            model_features.untyped_storage().data_ptr()
            != features.untyped_storage().data_ptr(),
            f"{condition.name} feature output has independent storage",
        )
        c.true(
            reconstruction_mask.untyped_storage().data_ptr()
            != target_mask.untyped_storage().data_ptr(),
            f"{condition.name} reconstruction mask has independent storage",
        )


def test_stochastic_conditions_do_not_advance_global_torch_rng(c: Check):
    features, valid_mask, target_mask = _synthetic_inputs()
    stochastic_conditions = BENCHMARK_CONDITIONS[3:]
    original_state = torch.random.get_rng_state()
    try:
        for condition in stochastic_conditions:
            torch.manual_seed(90210)
            state_before = torch.random.get_rng_state().clone()
            build_condition_inputs(features, valid_mask, target_mask, condition)
            state_after = torch.random.get_rng_state()
            c.true(torch.equal(state_after, state_before),
                   f"{condition.name} uses only its local CPU generator")
    finally:
        torch.random.set_rng_state(original_state)


def test_frame_shuffle_uses_the_validated_condition_seed(c: Check):
    source = inspect.getsource(build_condition_inputs)
    frame_branch = source.split(
        'elif condition.name == "frame_order_shuffle":', 1,
    )[1]
    c.true("assert condition.rng_seed is not None" in frame_branch,
           "the registered frame-shuffle seed is asserted")
    c.true("generator.manual_seed(condition.rng_seed)" in frame_branch,
           "frame shuffle seeds its local generator from the condition")
    c.true("generator.manual_seed(63000)" not in frame_branch,
           "frame shuffle does not duplicate the registered seed as a literal")


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


def _metric_bundle(value: float) -> dict:
    return {
        baseline: {
            "raw_mae": {
                "blendshape72": value,
                "clinical23": value,
                "equal_block_macro": value,
                "full95": value,
            },
            "standardized_mae": value,
            "standardized_smooth_l1": value,
        }
        for baseline in ("trained", "fresh_untrained", "train_mean")
    }


def _all_condition_rows(clean_values=(1.0, 3.0, 5.0)) -> list[dict]:
    rows = []
    for condition_index, condition in enumerate(BENCHMARK_CONDITIONS):
        for seed in range(3):
            value = clean_values[seed] if condition.name == "clean_fusion" else 2.0
            rows.append({
                "condition": condition.name,
                "seed": seed,
                "metrics": _metric_bundle(value),
            })
    return rows


def test_metric_bundle_canonicalizes_json_safe_values(c: Check):
    bundle = _metric_bundle(1.234565)
    normalized = validate_metric_bundle(bundle)
    c.eq(canonical_metric(1.234565), canonical_metric(1.23456),
         "metrics use banker's rounding at five decimal places")
    c.eq(normalized["trained"]["raw_mae"]["full95"], 1.23456,
         "bundle values are JSON-safe canonical floats")
    c.true(normalized is not bundle, "bundle validation returns fresh storage")
    for invalid in (True, "1.0", float("nan"), float("inf"), -0.001):
        c.raises(lambda invalid=invalid: canonical_metric(invalid),
                 (TypeError if type(invalid) in (bool, str) else ValueError),
                 "invalid metric scalars fail closed")
    malformed = _metric_bundle(1.0)
    malformed["trained"]["raw_mae"].pop("full95")
    c.raises(lambda: validate_metric_bundle(malformed), ValueError,
             "metric schemas require every exact key")


def test_clean_replay_requires_each_exact_seed_and_metric(c: Check):
    observed = [
        {"condition": "clean_fusion", "seed": seed, "metrics": _metric_bundle(seed + 1.0)}
        for seed in range(3)
    ]
    expected = {seed: _metric_bundle(seed + 1.0) for seed in range(3)}
    require_clean_replay(observed, expected)
    changed = [dict(row) for row in observed]
    changed[0] = dict(changed[0], metrics=_metric_bundle(9.0))
    c.raises(lambda: require_clean_replay(changed, expected), ValueError,
             "clean replay rejects any canonical metric mismatch")
    c.raises(lambda: require_clean_replay(observed[:2], expected), ValueError,
             "clean replay requires all three seed rows")


def test_clean_replay_rejects_hostile_expected_keys_before_hash_or_equality(c: Check):
    observed = [
        {"condition": "clean_fusion", "seed": seed, "metrics": _metric_bundle(1.0)}
        for seed in range(3)
    ]
    hostile = _HostileKey(0)
    expected = {hostile: _metric_bundle(1.0), 1: _metric_bundle(1.0), 2: _metric_bundle(1.0)}
    hostile.hashes = 0
    hostile.comparisons = 0
    c.raises(lambda: require_clean_replay(observed, expected), ValueError,
             "expected seed keys require exact int type before set operations")
    c.eq(hostile.hashes, 0, "validation never hashes a hostile expected seed key")
    c.eq(hostile.comparisons, 0, "validation never compares a hostile expected seed key")


def test_aggregate_conditions_enforces_complete_grid_and_computes_stats(c: Check):
    rows = _all_condition_rows()
    report = aggregate_condition_metrics(rows)
    clean = report["conditions"][0]
    metric = clean["aggregates"]["trained"]["raw_mae"]["equal_block_macro"]
    c.eq(metric, {"mean": 3.0, "sample_sd": 2.0},
         "condition aggregates use arithmetic mean and sample standard deviation")
    c.eq(clean["degradation_percent_vs_clean"], 0.0,
         "clean condition degradation is exactly zero")
    c.eq(report["conditions"][1]["degradation_percent_vs_clean"], 100.0 * (2.0 / 3.0 - 1.0),
         "degradation is computed against the clean trained macro mean")
    c.eq(len(report["conditions"]), 10, "report contains only registered conditions")
    c.raises(lambda: aggregate_condition_metrics(rows[:-1]), ValueError,
             "missing grid cells fail closed")
    duplicate = rows + [rows[0]]
    c.raises(lambda: aggregate_condition_metrics(duplicate), ValueError,
             "duplicate grid cells fail closed")
    extra = list(rows)
    extra[-1] = dict(extra[-1], condition="unregistered")
    c.raises(lambda: aggregate_condition_metrics(extra), ValueError,
             "extra condition names fail closed")


def test_aggregate_conditions_rejects_bad_rows_and_keeps_recordings_out(c: Check):
    rows = _all_condition_rows()
    bad_type = list(rows)
    bad_type[0] = dict(bad_type[0], seed=True)
    c.raises(lambda: aggregate_condition_metrics(bad_type), ValueError,
             "seed must have exact integer type")
    recording = list(rows)
    recording[0] = dict(recording[0], recording_id="rec_secret")
    c.raises(lambda: aggregate_condition_metrics(recording), ValueError,
             "per-recording fields are never accepted")
    report = aggregate_condition_metrics(rows)
    c.true("recording_id" not in repr(report),
           "aggregate reports retain no per-recording data")


def test_deidentified_payload_accepts_only_closed_aggregate_schema(c: Check):
    safe = aggregate_condition_metrics(_all_condition_rows())
    normalized = validate_deidentified_payload(safe)
    c.eq(normalized, safe, "the exact aggregate result is accepted")
    c.true(normalized is not safe and normalized["conditions"] is not safe["conditions"],
           "validation reconstructs fresh allowlisted storage")

    for forbidden_key in (
        "patient_id", "subject_id", "recording_id", "private_key", "path", "extra",
    ):
        unsafe = {"conditions": list(safe["conditions"]), forbidden_key: "secret"}
        c.raises(lambda unsafe=unsafe: validate_deidentified_payload(unsafe), ValueError,
                 "every non-allowlisted top-level field fails closed")

    wrong_condition = {"conditions": [dict(item) for item in safe["conditions"]]}
    wrong_condition["conditions"][0]["condition"] = "wrong"
    c.raises(lambda: validate_deidentified_payload(wrong_condition), ValueError,
             "condition identity and order are exact")

    nested_extra = {"conditions": [dict(item) for item in safe["conditions"]]}
    nested_extra["conditions"][0]["patient_id"] = "patient_1"
    c.raises(lambda: validate_deidentified_payload(nested_extra), ValueError,
             "condition entries reject private and extra fields")

    non_public_metric = {"conditions": [dict(item) for item in safe["conditions"]]}
    first = non_public_metric["conditions"][0]
    first["seed_rows"] = [dict(row) for row in first["seed_rows"]]
    first["seed_rows"][0] = dict(first["seed_rows"][0])
    first["seed_rows"][0]["metrics"] = _metric_bundle(Decimal("1.0"))
    c.raises(lambda: validate_deidentified_payload(non_public_metric), ValueError,
             "the publication boundary requires exact JSON float metric values")

    out_of_range_summary = {"conditions": [dict(item) for item in safe["conditions"]]}
    first = out_of_range_summary["conditions"][0]
    first["aggregates"] = {
        baseline: {
            metric: (dict(value) if type(value) is dict else value)
            for metric, value in metrics.items()
        }
        for baseline, metrics in first["aggregates"].items()
    }
    first["aggregates"]["trained"]["standardized_mae"]["mean"] = 1_000_000_000.00001
    c.raises(lambda: validate_deidentified_payload(out_of_range_summary), ValueError,
             "aggregate metric summaries enforce the publication metric bound")

    cyclic = {}
    cyclic["conditions"] = cyclic
    c.raises(lambda: validate_deidentified_payload(cyclic), ValueError,
             "cyclic arbitrary structures fail through the schema, not recursion")
    deep = value = []
    for _ in range(2000):
        child = []
        value.append(child)
        value = child
    c.raises(lambda: validate_deidentified_payload({"conditions": deep}), ValueError,
             "deep arbitrary structures fail without recursive traversal")


def test_json_boundary_preserves_distinct_near_limit_canonical_metrics(c: Check):
    observed = [
        {
            "condition": "clean_fusion",
            "seed": seed,
            "metrics": validate_metric_bundle(
                _metric_bundle(Decimal("999999999.99998"))
            ),
        }
        for seed in range(3)
    ]
    expected = {
        seed: _metric_bundle(Decimal("999999999.99998"))
        for seed in range(3)
    }
    expected[1] = _metric_bundle(Decimal("999999999.99999"))
    c.raises(lambda: require_clean_replay(observed, expected), ValueError,
             "JSON composition preserves distinct bounded five-place metrics")


def test_aggregation_preserves_tiny_large_decimal_differences(c: Check):
    values = tuple(Decimal("999999999.99997") + Decimal(seed) / Decimal("100000")
                   for seed in range(3))
    rows = _all_condition_rows(values)
    report = aggregate_condition_metrics(rows)
    metric = report["conditions"][0]["aggregates"]["trained"]["raw_mae"]["full95"]
    c.true(metric["sample_sd"] > 0.0,
           "sample deviation retains canonical differences beyond float resolution")
    c.eq(metric["sample_sd"], float(Decimal("0.00001")),
         "Decimal sample deviation is computed before JSON conversion")


def test_canonical_metric_enforces_bound_and_representation_limits(c: Check):
    for invalid in (
        Decimal("1000000000.00001"),
        Decimal("1e27"),
        Decimal("0e101"),
        Decimal("0." + "1234567890" * 7),
    ):
        c.raises(lambda invalid=invalid: canonical_metric(invalid), ValueError,
                 "out-of-domain, extreme-exponent, and overlong metrics fail closed")


def test_canonical_metric_uses_private_context(c: Check):
    original = getcontext().copy()
    try:
        ambient = getcontext()
        ambient.prec = 2
        ambient.Emax = 2
        ambient.Emin = -2
        ambient.traps[Inexact] = True
        ambient.traps[Rounded] = True
        c.eq(canonical_metric(Decimal("1.234565")), Decimal("1.23456"),
             "ambient precision, exponent bounds, and traps cannot alter results")
    finally:
        setcontext(original)


def test_every_numeric_zero_is_normalized_to_positive_zero(c: Check):
    for zero in (0, -0.0, Decimal("-0")):
        canonical = canonical_metric(zero)
        c.true(canonical.is_zero() and not canonical.is_signed(),
               "canonical metrics normalize every signed-zero representation")

    rows = _all_condition_rows()
    for row in rows:
        if row["condition"] == "mask_landmarks":
            row["metrics"] = _metric_bundle(-0.0)
    report = aggregate_condition_metrics(rows)
    report["conditions"][0]["aggregates"]["trained"]["standardized_mae"]["sample_sd"] = -0.0
    report["conditions"][0]["degradation_percent_vs_clean"] = -0.0
    normalized = validate_deidentified_payload(report)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    c.true("-0.0" not in encoded,
           "valid public aggregate JSON never contains negative zero bytes")


def test_metric_schema_rejects_non_exact_or_hostile_keys_before_comparison(c: Check):
    subclassed = _metric_bundle(1.0)
    subclassed[_StrSubclass("trained")] = subclassed.pop("trained")
    c.raises(lambda: validate_metric_bundle(subclassed), ValueError,
             "str subclasses are not exact schema keys")

    hostile = _HostileKey("trained")
    hostile_bundle = _metric_bundle(1.0)
    hostile_bundle[hostile] = hostile_bundle.pop("trained")
    c.raises(lambda: validate_metric_bundle(hostile_bundle), ValueError,
             "hostile equality keys fail before schema comparison")
    c.eq(hostile.comparisons, 0, "hostile key equality is never invoked")


def _fusion_inference_fixture():
    """Build a small, real Mayo SSL heldout partition without external data."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(701)
    raw_features = torch.randn(3, 4, 32, 95, generator=generator)
    valid_mask = torch.ones((3, 4, 32), dtype=torch.bool)
    split = ssl_core.SSLGroupSplit(
        train_indices=np.asarray([0], dtype=np.int64),
        heldout_indices=np.asarray([1, 2], dtype=np.int64),
        unit="recording",
        claim_unit="recording_held_out_not_patient_held_out",
        patient_held_out=False,
    )
    groups = ("train", "heldout_a", "heldout_b")
    scaler = ssl_core.fit_source_scaler(
        raw_features, valid_mask,
        source=ssl_core.MAYO_SOURCE,
        fit_indices=split.train_indices,
        heldout_indices=split.heldout_indices,
    )
    features = scaler.transform(raw_features, valid_mask, source=ssl_core.MAYO_SOURCE)[1:]
    valid_mask = valid_mask[1:]
    target_mask = torch.zeros_like(valid_mask)
    target_mask[:, :, 8:12] = True
    timestamps = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32).repeat(2, 4, 1) / 30.0
    source_indices = torch.arange(32, dtype=torch.int64).reshape(1, 1, 32).repeat(2, 4, 1)

    def model(seed: int):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return ssl_core.DynamicLandmarkSSLModel()

    trained = {seed: model(800 + seed) for seed in range(3)}
    fresh = {seed: model(900 + seed) for seed in range(3)}
    return {
        "trained_models": trained,
        "fresh_models": fresh,
        "features": features,
        "valid_mask": valid_mask,
        "timestamps": timestamps,
        "source_frame_indices": source_indices,
        "target_mask": target_mask,
        "scaler": scaler,
        "split": split,
        "evaluated_indices": split.heldout_indices,
        "group_ids": groups,
    }


def _expected_clean_metrics(fixture):
    expected = {}
    for seed in range(3):
        trained = fixture["trained_models"][seed](
            fixture["features"], fixture["valid_mask"], fixture["timestamps"],
            fixture["source_frame_indices"], reconstruction_mask=fixture["target_mask"],
            source="mayo", input_arm="fusion",
        )
        fresh = fixture["fresh_models"][seed](
            fixture["features"], fixture["valid_mask"], fixture["timestamps"],
            fixture["source_frame_indices"], reconstruction_mask=fixture["target_mask"],
            source="mayo", input_arm="fusion",
        )
        report = ssl_core.reconstruction_report(
            trained, fresh, fixture["features"], fixture["target_mask"],
            baseline=fixture["scaler"], split=fixture["split"],
            evaluated_indices=fixture["evaluated_indices"], group_ids=fixture["group_ids"],
            source=ssl_core.MAYO_SOURCE,
        )
        metrics = report["common_target_metrics"]
        expected[seed] = validate_metric_bundle({
            "trained": metrics["trained"],
            "fresh_untrained": metrics["untrained"],
            "train_mean": metrics["train_mean"],
        })
    return expected


def test_real_model_inference_returns_frozen_deidentified_grid(c: Check):
    fixture = _fusion_inference_fixture()
    expected = _expected_clean_metrics(fixture)
    fixture["trained_models"][0].train()
    fixture["fresh_models"][1].train()
    model_state = {
        id(model): {name: value.detach().clone() for name, value in model.state_dict().items()}
        for mapping in (fixture["trained_models"], fixture["fresh_models"])
        for model in mapping.values()
    }
    rng_before = torch.random.get_rng_state().clone()
    rows = evaluate_fusion_conditions(
        **fixture, expected_clean_metrics_by_seed=expected,
    )
    c.eq(len(rows), 30, "every registered condition has every seed")
    c.eq(
        [(row["condition"], row["seed"]) for row in rows],
        [(condition.name, seed) for condition in BENCHMARK_CONDITIONS for seed in range(3)],
        "rows are condition-major in frozen condition and seed order",
    )
    c.eq(rows[:3], [
        {"condition": "clean_fusion", "seed": seed, "metrics": expected[seed]}
        for seed in range(3)
    ], "clean metrics use the exact common clean target report")
    c.true(all(set(row) == {"condition", "seed", "metrics"} for row in rows),
           "rows contain no identifiers or per-recording output")
    c.true(validate_deidentified_payload(aggregate_condition_metrics(rows)) is not None,
           "returned rows are accepted by the deidentified publication boundary")
    c.true(torch.equal(torch.random.get_rng_state(), rng_before),
           "inference and deterministic perturbations do not mutate global RNG")
    for mapping in (fixture["trained_models"], fixture["fresh_models"]):
        for model in mapping.values():
            c.true(all(torch.equal(value, model.state_dict()[name])
                       for name, value in model_state[id(model)].items()),
                   "real model parameters and buffers remain unchanged")
    c.true(fixture["trained_models"][0].training,
           "caller training modes are restored after deterministic eval inference")
    c.true(fixture["fresh_models"][1].training,
           "fresh caller training modes are restored after deterministic eval inference")


def test_clean_replay_failure_stops_before_stress_for_real_models(c: Check):
    fixture = _fusion_inference_fixture()
    expected = deepcopy(_expected_clean_metrics(fixture))
    expected[0]["trained"]["standardized_mae"] += 0.00001
    calls = [0]
    hooks = [model.register_forward_hook(lambda *_args: calls.__setitem__(0, calls[0] + 1))
             for mapping in (fixture["trained_models"], fixture["fresh_models"])
             for model in mapping.values()]
    try:
        c.raises(lambda: evaluate_fusion_conditions(
            **fixture, expected_clean_metrics_by_seed=expected,
        ), ValueError, "a mismatched clean replay fails closed")
    finally:
        for hook in hooks:
            hook.remove()
    c.eq(calls[0], 6,
         "only the three clean trained/fresh pairs run before the replay gate")


def test_stress_conditions_keep_the_clean_target_and_scoring_mask(c: Check):
    fixture = _fusion_inference_fixture()
    expected = _expected_clean_metrics(fixture)
    observed = []
    original = ssl_core.reconstruction_report

    def capture_target_and_mask(trained, fresh, target, reconstruction_mask, **kwargs):
        observed.append((target, reconstruction_mask))
        return original(trained, fresh, target, reconstruction_mask, **kwargs)

    ssl_core.reconstruction_report = capture_target_and_mask
    try:
        evaluate_fusion_conditions(**fixture, expected_clean_metrics_by_seed=expected)
    finally:
        ssl_core.reconstruction_report = original
    c.eq(len(observed), 30, "every real-model condition is scored once per seed")
    c.true(all(target is fixture["features"] and mask is fixture["target_mask"]
               for target, mask in observed),
           "dropout and noise score only the original clean target positions")


def test_inference_rejects_nonexact_model_maps_and_temporal_types(c: Check):
    fixture = _fusion_inference_fixture()
    expected = _expected_clean_metrics(fixture)
    hostile = _HostileKey(0)
    hostile_models = dict(fixture["trained_models"])
    hostile_models.pop(0)
    hostile_models[hostile] = fixture["trained_models"][1]
    c.raises(lambda: evaluate_fusion_conditions(
        **{**fixture, "trained_models": hostile_models},
        expected_clean_metrics_by_seed=expected,
    ), ValueError, "hostile model keys fail before equality")
    c.eq(hostile.comparisons, 0, "hostile model-key equality is never invoked")
    c.raises(lambda: evaluate_fusion_conditions(
        **{**fixture, "timestamps": fixture["timestamps"].to(torch.int64)},
        expected_clean_metrics_by_seed=expected,
    ), ValueError, "timestamps require finite floating tensor provenance")
    c.raises(lambda: evaluate_fusion_conditions(
        **{**fixture, "source_frame_indices": fixture["source_frame_indices"].to(torch.int32)},
        expected_clean_metrics_by_seed=expected,
    ), ValueError, "source frame indices require exact int64 provenance")


if __name__ == "__main__":
    run_all("test_focused_fusion_robustness", dict(globals()))
