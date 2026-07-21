"""Protocol-contract tests for the focused Fusion robustness benchmark."""
from __future__ import annotations

import inspect
import importlib.util
import io
import json
import stat
import sys
import tempfile
import threading
from types import SimpleNamespace
from contextlib import redirect_stdout
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
from src.evaluation import focused_fusion_robustness as fusion_core  # noqa: E402


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
    models = tuple((
        *fixture["trained_models"].values(), *fixture["fresh_models"].values(),
    ))
    module_modes = tuple(
        (module, module.training) for model in models for module in model.modules()
    )
    try:
        for model in models:
            model.eval()
        expected = {}
        with torch.no_grad():
            for seed in range(3):
                trained = fixture["trained_models"][seed](
                    fixture["features"], fixture["valid_mask"], fixture["timestamps"],
                    fixture["source_frame_indices"],
                    reconstruction_mask=fixture["target_mask"],
                    source="mayo", input_arm="fusion",
                )
                fresh = fixture["fresh_models"][seed](
                    fixture["features"], fixture["valid_mask"], fixture["timestamps"],
                    fixture["source_frame_indices"],
                    reconstruction_mask=fixture["target_mask"],
                    source="mayo", input_arm="fusion",
                )
                report = ssl_core.reconstruction_report(
                    trained, fresh, fixture["features"], fixture["target_mask"],
                    baseline=fixture["scaler"], split=fixture["split"],
                    evaluated_indices=fixture["evaluated_indices"],
                    group_ids=fixture["group_ids"], source=ssl_core.MAYO_SOURCE,
                )
                metrics = report["common_target_metrics"]
                expected[seed] = validate_metric_bundle({
                    "trained": metrics["trained"],
                    "fresh_untrained": metrics["untrained"],
                    "train_mean": metrics["train_mean"],
                })
        return expected
    finally:
        for module, mode in module_modes:
            module.training = mode


def test_expected_clean_helper_uses_eval_no_grad_and_restores_nested_modes(c: Check):
    fixture = _fusion_inference_fixture()
    models = tuple((
        *fixture["trained_models"].values(), *fixture["fresh_models"].values(),
    ))
    modes_before = {}
    for model_number, model in enumerate(models):
        for module_number, module in enumerate(model.modules()):
            module.training = bool((model_number + module_number) % 2)
            modes_before[id(module)] = module.training
    observations = []
    hooks = [model.register_forward_pre_hook(
        lambda observed_model, _args: observations.append((
            torch.is_grad_enabled(),
            tuple(module.training for module in observed_model.modules()),
        )),
    ) for model in models]
    try:
        _expected_clean_metrics(fixture)
    finally:
        for hook in hooks:
            hook.remove()
    c.eq(len(observations), 6, "helper evaluates one trained/fresh pair per seed")
    c.true(all(not grad_enabled and not any(module_modes)
               for grad_enabled, module_modes in observations),
           "helper clean forwards use eval mode with gradients disabled")
    c.eq(
        {id(module): module.training for model in models for module in model.modules()},
        modes_before,
        "helper restores every heterogeneous nested module mode",
    )


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
    models = tuple((
        *fixture["trained_models"].values(), *fixture["fresh_models"].values(),
    ))
    for model_number, model in enumerate(models):
        for module_number, module in enumerate(model.modules()):
            module.training = bool((model_number + module_number) % 2)
    modes_before = {
        id(module): module.training for model in models for module in model.modules()
    }
    states_before = {
        id(model): {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        for model in models
    }
    rng_before = torch.random.get_rng_state().clone()
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
    c.eq(
        {id(module): module.training for model in models for module in model.modules()},
        modes_before,
        "clean replay exceptions restore every nested module mode",
    )
    c.true(all(
        torch.equal(value, model.state_dict()[name])
        for model in models
        for name, value in states_before[id(model)].items()
    ), "clean replay exceptions preserve every parameter and buffer")
    c.true(torch.equal(torch.random.get_rng_state(), rng_before),
           "clean replay exceptions preserve global RNG state")


def test_clean_replay_uses_original_inputs_without_condition_builder(c: Check):
    fixture = _fusion_inference_fixture()
    expected = deepcopy(_expected_clean_metrics(fixture))
    expected[0]["trained"]["standardized_mae"] += 0.00001
    builder_calls = []
    original_builder = fusion_core.build_condition_inputs

    def capture_builder(*args, **kwargs):
        builder_calls.append(True)
        return original_builder(*args, **kwargs)

    forward_inputs = []
    hooks = [model.register_forward_pre_hook(
        lambda _model, args, kwargs: forward_inputs.append(
            (args[0], kwargs["reconstruction_mask"]),
        ),
        with_kwargs=True,
    ) for mapping in (fixture["trained_models"], fixture["fresh_models"])
        for model in mapping.values()]
    fusion_core.build_condition_inputs = capture_builder
    try:
        c.raises(lambda: evaluate_fusion_conditions(
            **fixture, expected_clean_metrics_by_seed=expected,
        ), ValueError, "the deliberately mismatched replay still reaches the gate")
    finally:
        fusion_core.build_condition_inputs = original_builder
        for hook in hooks:
            hook.remove()
    c.eq(builder_calls, [], "clean replay invokes no condition builder before failing")
    c.eq(len(forward_inputs), 6, "only clean model pairs run before the gate")
    c.true(all(
        features is fixture["features"] and mask is fixture["target_mask"]
        for features, mask in forward_inputs
    ), "clean inference receives the caller's exact tensors")


def test_inference_restores_each_nested_module_training_flag(c: Check):
    fixture = _fusion_inference_fixture()
    expected = _expected_clean_metrics(fixture)
    before = {}
    for model_number, model in enumerate(
        (*fixture["trained_models"].values(), *fixture["fresh_models"].values())
    ):
        for module_number, module in enumerate(model.modules()):
            module.training = bool((model_number + module_number) % 2)
            before[id(module)] = module.training
    evaluate_fusion_conditions(**fixture, expected_clean_metrics_by_seed=expected)
    after = {
        id(module): module.training
        for model in (*fixture["trained_models"].values(), *fixture["fresh_models"].values())
        for module in model.modules()
    }
    c.eq(after, before,
         "eval inference restores caller-provided heterogeneous nested module modes")


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


def _assert_model_mapping_rejected_before_inference(c: Check, fixture, **overrides):
    calls = [0]
    unique_models = {
        id(model): model
        for mapping in (fixture["trained_models"], fixture["fresh_models"])
        for model in mapping.values()
    }
    hooks = [model.register_forward_hook(
        lambda *_args: calls.__setitem__(0, calls[0] + 1),
    ) for model in unique_models.values()]
    try:
        c.raises(lambda: evaluate_fusion_conditions(
            **{**fixture, **overrides}, expected_clean_metrics_by_seed={},
        ), ValueError, "invalid model mappings fail closed")
    finally:
        for hook in hooks:
            hook.remove()
    c.eq(calls[0], 0, "invalid model mappings are rejected before any inference")


def test_trained_seed_models_require_distinct_root_identities(c: Check):
    fixture = _fusion_inference_fixture()
    trained = dict(fixture["trained_models"])
    trained[1] = trained[0]
    _assert_model_mapping_rejected_before_inference(
        c, fixture, trained_models=trained,
    )


def test_fresh_seed_models_require_distinct_root_identities(c: Check):
    fixture = _fusion_inference_fixture()
    fresh = dict(fixture["fresh_models"])
    fresh[2] = fresh[0]
    _assert_model_mapping_rejected_before_inference(
        c, fixture, fresh_models=fresh,
    )


def test_trained_and_fresh_maps_cannot_share_root_models(c: Check):
    fixture = _fusion_inference_fixture()
    fresh = dict(fixture["fresh_models"])
    fresh[2] = fixture["trained_models"][1]
    _assert_model_mapping_rejected_before_inference(
        c, fixture, fresh_models=fresh,
    )


def test_model_maps_require_exact_real_model_types(c: Check):
    fixture = _fusion_inference_fixture()

    class ModelSubclass(ssl_core.DynamicLandmarkSSLModel):
        pass

    with torch.random.fork_rng(devices=[]):
        subclassed = ModelSubclass()
    for invalid in (subclassed, torch.nn.Linear(1, 1)):
        trained = dict(fixture["trained_models"])
        trained[0] = invalid
        _assert_model_mapping_rejected_before_inference(
            c, fixture, trained_models=trained,
        )


def test_model_maps_reject_nonfinite_parameter_and_buffer_state(c: Check):
    parameter_fixture = _fusion_inference_fixture()
    parameter = next(parameter_fixture["trained_models"][0].parameters())
    with torch.no_grad():
        parameter.reshape(-1)[0] = float("nan")
    _assert_model_mapping_rejected_before_inference(c, parameter_fixture)

    buffer_fixture = _fusion_inference_fixture()
    buffer_fixture["fresh_models"][2].register_buffer(
        "test_nonfinite_buffer", torch.tensor(float("inf")),
    )
    _assert_model_mapping_rejected_before_inference(c, buffer_fixture)


def test_model_maps_require_cpu_model_state(c: Check):
    fixture = _fusion_inference_fixture()
    with torch.random.fork_rng(devices=[]):
        non_cpu = ssl_core.DynamicLandmarkSSLModel().to(device="meta")
    trained = dict(fixture["trained_models"])
    trained[2] = non_cpu
    _assert_model_mapping_rejected_before_inference(
        c, fixture, trained_models=trained,
    )


def _load_cli():
    script = ROOT / "scripts" / "run_focused_fusion_robustness.py"
    spec = importlib.util.spec_from_file_location(
        "focused_fusion_robustness_cli", script,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("focused Fusion CLI cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_public_report(cli) -> dict:
    aggregate = aggregate_condition_metrics(_all_condition_rows())
    return {
        "schema_version": "focused_fusion_robustness_report_v1",
        "status": "complete",
        "claim_scope": "recording_heldout_development_reconstruction_stress_only",
        "source": "mayo",
        "selected_arm": "fusion",
        "seeds": [0, 1, 2],
        "metric_policy": {
            "canonicalization": "decimal_round_half_even_v1",
            "decimal_places": 5,
            "input_metric_min": 0.0,
            "input_metric_max": 1e9,
            "input_metric_max_digits": 64,
            "input_metric_exponent_min": -100,
            "input_metric_exponent_max": 100,
            "primary_metric": "trained.raw_mae.equal_block_macro",
            "lower_is_better": True,
            "degradation_formula": "100*(condition_mean/clean_mean-1)",
            "degradation_range": [-100.0, 1e16],
        },
        "accounting": {
            "heldout_packets": 160,
            "heldout_recording_groups": 10,
            "valid_positions": 20434,
            "scored_target_positions": 5120,
            "scored_target_scalars": 486400,
            "observed_context_positions": 15314,
            "feature_width": 95,
        },
        "protocol_registry": [
            {
                "name": condition.name,
                "input_arm": condition.input_arm,
                "context_dropout_probability": condition.context_dropout_probability,
                "landmark_noise_sd": condition.landmark_noise_sd,
                "rng_seed": condition.rng_seed,
            }
            for condition in BENCHMARK_CONDITIONS
        ],
        "commitments": {
            "benchmark_script_sha256": "1" * 64,
            "evaluation_module_sha256": "2" * 64,
            "trainer_sha256": "3" * 64,
            "bridge_generation_sha256": "4" * 64,
            "common_contract_sha256": "5" * 64,
            "winner_report_sha256": "6" * 64,
            "checkpoints": [
                {
                    "seed": seed,
                    "checkpoint_fingerprint": str(7 + seed) * 64,
                    "checkpoint_receipt_sha256": chr(ord("a") + seed) * 64,
                }
                for seed in range(3)
            ],
        },
        "conditions": aggregate["conditions"],
    }


def _replace_exact_key(mapping: dict, name: str, replacement: object) -> object:
    value = mapping.pop(name)
    mapping[replacement] = value
    return replacement


def _public_mapping_layers(report: dict) -> tuple[tuple[str, dict, str], ...]:
    return (
        ("top", report, "status"),
        ("metric_policy", report["metric_policy"], "decimal_places"),
        ("accounting", report["accounting"], "heldout_packets"),
        ("protocol", report["protocol_registry"][0], "name"),
        ("commitments", report["commitments"], "trainer_sha256"),
        (
            "checkpoint",
            report["commitments"]["checkpoints"][0],
            "checkpoint_fingerprint",
        ),
    )


def test_cli_is_import_safe_and_accepts_only_the_fixed_no_argument_job(c: Check):
    cli = _load_cli()
    c.eq(cli._parser().parse_args([]).__dict__, {},
         "the fixed benchmark parser has an exact empty namespace")
    c.raises(lambda: cli._parser().parse_args(["--output", "elsewhere"]),
             SystemExit, "callers cannot redirect or customize the benchmark")
    c.eq(
        cli.DEFAULT_REPORT_PATH,
        ROOT / "outputs" / "dynamic_landmark" / "benchmarks"
        / "development" / "focused-fusion-robustness-v1" / "report.json",
        "the no-argument publication path is exact",
    )


def test_full_report_validator_reconstructs_only_the_exact_public_schema(c: Check):
    cli = _load_cli()
    report = _valid_public_report(cli)
    normalized = cli.validate_public_report(report)
    c.eq(normalized, report, "the exact public report is accepted")
    c.true(normalized is not report and normalized["conditions"] is not report["conditions"],
           "the validator reconstructs fresh allowlisted storage")
    c.eq(tuple(normalized), tuple(report), "top-level field order is frozen")
    c.eq(tuple(normalized["metric_policy"]), tuple(report["metric_policy"]),
         "metric-policy field order is frozen")
    c.eq(tuple(normalized["commitments"]), tuple(report["commitments"]),
         "commitment field order is frozen")


def test_cli_owned_mapping_layers_reject_hostile_keys_without_executing_them(c: Check):
    cli = _load_cli()
    for label in (
        "top", "metric_policy", "accounting", "protocol", "commitments",
        "checkpoint",
    ):
        report = _valid_public_report(cli)
        layer = next(item for item in _public_mapping_layers(report)
                     if item[0] == label)
        hostile = _HostileKey(layer[2])
        _replace_exact_key(layer[1], layer[2], hostile)
        hostile.hashes = 0
        hostile.comparisons = 0
        c.raises(
            lambda report=report: cli.validate_public_report(report),
            ValueError,
            f"{label} rejects a non-string key before hashing or equality",
        )
        c.eq(hostile.hashes, 0, f"{label} does not hash caller key code")
        c.eq(hostile.comparisons, 0,
             f"{label} does not compare caller key code")


def test_cli_owned_mapping_layers_reject_string_subclass_keys(c: Check):
    cli = _load_cli()
    for label in (
        "top", "metric_policy", "accounting", "protocol", "commitments",
        "checkpoint",
    ):
        report = _valid_public_report(cli)
        layer = next(item for item in _public_mapping_layers(report)
                     if item[0] == label)
        _replace_exact_key(layer[1], layer[2], _StrSubclass(layer[2]))
        c.raises(
            lambda report=report: cli.validate_public_report(report),
            ValueError,
            f"{label} accepts only exact built-in string keys",
        )


def test_public_report_recomputes_every_aggregate_and_degradation(c: Check):
    cli = _load_cli()
    mutations = []
    seed_row = _valid_public_report(cli)
    seed_row["conditions"][0]["seed_rows"][0]["metrics"]["trained"][
        "raw_mae"
    ]["equal_block_macro"] = 1.25
    mutations.append(("seed row", seed_row))
    mean = _valid_public_report(cli)
    mean["conditions"][0]["aggregates"]["trained"]["raw_mae"][
        "equal_block_macro"
    ]["mean"] = 3.25
    mutations.append(("mean", mean))
    sample_sd = _valid_public_report(cli)
    sample_sd["conditions"][0]["aggregates"]["trained"]["raw_mae"][
        "equal_block_macro"
    ]["sample_sd"] = 2.25
    mutations.append(("sample standard deviation", sample_sd))
    clean_degradation = _valid_public_report(cli)
    clean_degradation["conditions"][0]["degradation_percent_vs_clean"] = 0.01
    mutations.append(("clean degradation", clean_degradation))
    nonclean_degradation = _valid_public_report(cli)
    nonclean_degradation["conditions"][1][
        "degradation_percent_vs_clean"
    ] += 0.01
    mutations.append(("non-clean degradation", nonclean_degradation))
    for label, report in mutations:
        c.raises(
            lambda report=report: cli.validate_public_report(report),
            ValueError,
            f"{label} must exactly match recomputation from all 30 seed rows",
        )


def test_full_report_validator_rejects_extras_identifiers_paths_and_secrets(c: Check):
    cli = _load_cli()
    for forbidden_key in (
        "extra", "patient_id", "recording_id", "path", "authority_hmac",
        "private_key",
    ):
        report = _valid_public_report(cli)
        report[forbidden_key] = "secret"
        c.raises(lambda report=report: cli.validate_public_report(report),
                 ValueError, f"{forbidden_key} is outside the publication allowlist")
    nested = _valid_public_report(cli)
    nested["commitments"]["checkpoints"][0]["checkpoint_path"] = "/private/model.pt"
    c.raises(lambda: cli.validate_public_report(nested), ValueError,
             "checkpoint paths cannot enter the public report")


def test_full_report_validator_rejects_wrong_literals_hashes_and_accounting(c: Check):
    cli = _load_cli()
    mutations = []
    wrong_arm = _valid_public_report(cli)
    wrong_arm["selected_arm"] = "landmark_only"
    mutations.append(wrong_arm)
    wrong_seeds = _valid_public_report(cli)
    wrong_seeds["seeds"] = [0, 1]
    mutations.append(wrong_seeds)
    wrong_accounting = _valid_public_report(cli)
    wrong_accounting["accounting"]["heldout_packets"] = 159
    mutations.append(wrong_accounting)
    wrong_hash = _valid_public_report(cli)
    wrong_hash["commitments"]["trainer_sha256"] = "A" * 64
    mutations.append(wrong_hash)
    wrong_checkpoint_hash = _valid_public_report(cli)
    wrong_checkpoint_hash["commitments"]["checkpoints"][1][
        "checkpoint_receipt_sha256"
    ] = "short"
    mutations.append(wrong_checkpoint_hash)
    for mutation in mutations:
        c.raises(lambda mutation=mutation: cli.validate_public_report(mutation),
                 ValueError, "wrong literal, accounting, or hash shape fails closed")


def test_full_report_validator_requires_exact_protocol_and_three_checkpoint_rows(c: Check):
    cli = _load_cli()
    wrong_protocol = _valid_public_report(cli)
    wrong_protocol["protocol_registry"] = list(reversed(
        wrong_protocol["protocol_registry"],
    ))
    c.raises(lambda: cli.validate_public_report(wrong_protocol), ValueError,
             "protocol rows require frozen order and exact values")
    wrong_checkpoints = _valid_public_report(cli)
    wrong_checkpoints["commitments"]["checkpoints"] = wrong_checkpoints[
        "commitments"
    ]["checkpoints"][:2]
    c.raises(lambda: cli.validate_public_report(wrong_checkpoints), ValueError,
             "all and only three seed commitments are required")


def test_full_report_validation_delegates_to_closed_aggregate_validator(c: Check):
    cli = _load_cli()
    calls = []
    original = cli.fusion_core.validate_deidentified_payload

    def reject(_payload):
        calls.append(True)
        raise ValueError("closed aggregate rejection")

    cli.fusion_core.validate_deidentified_payload = reject
    try:
        c.raises(lambda: cli.validate_public_report(_valid_public_report(cli)),
                 ValueError, "aggregate validation remains fail closed")
    finally:
        cli.fusion_core.validate_deidentified_payload = original
    c.eq(calls, [True], "the existing closed aggregate validator is authoritative")


def test_canonical_json_is_deterministic_compact_ascii_and_rejects_nan(c: Check):
    cli = _load_cli()
    left = {"z": "caf\u00e9", "a": [1, 2.0]}
    right = {"a": [1, 2.0], "z": "caf\u00e9"}
    expected = b'{"a":[1,2.0],"z":"caf\\u00e9"}'
    c.eq(cli.canonical_json_bytes(left), expected,
         "canonical JSON is sorted, compact, and ASCII")
    c.eq(cli.canonical_json_bytes(left), cli.canonical_json_bytes(right),
         "mapping insertion order cannot change report bytes")
    c.raises(lambda: cli.canonical_json_bytes({"bad": float("nan")}),
             ValueError, "NaN is never serializable")


def test_private_atomic_report_write_enforces_modes_replace_and_symlink_rejection(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private_root = root / "benchmarks"
        report_path = private_root / "development" / "fixed" / "report.json"
        first = b'{"status":"first"}'
        second = b'{"status":"second"}'
        cli._atomic_write_report(report_path, first, private_root=private_root)
        c.eq(report_path.read_bytes(), first, "the first report is durable")
        c.eq(stat.S_IMODE(report_path.stat().st_mode), 0o600,
             "the report is owner-only")
        for directory in (private_root, report_path.parent.parent, report_path.parent):
            c.eq(stat.S_IMODE(directory.stat().st_mode), 0o700,
                 "every benchmark directory is owner-only")
        cli._atomic_write_report(report_path, second, private_root=private_root)
        c.eq(report_path.read_bytes(), second,
             "a repeated run atomically replaces only the fixed report")

        report_path.unlink()
        target = root / "outside.json"
        target.write_bytes(b"outside")
        report_path.symlink_to(target)
        c.raises(
            lambda: cli._atomic_write_report(
                report_path, first, private_root=private_root,
            ),
            ValueError,
            "an existing symlink can never be replaced",
        )
        c.eq(target.read_bytes(), b"outside", "symlink target remains untouched")


def test_private_publisher_rejects_unsafe_anchor_parent_and_lock(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private_root = root / "benchmarks"
        report_path = private_root / "development" / "fixed" / "report.json"
        root.chmod(0o777)
        c.raises(
            lambda: cli._atomic_write_report(
                report_path, b'{"status":"unsafe-parent"}',
                private_root=private_root,
            ),
            ValueError,
            "a group/world-writable private-root parent is rejected",
        )
        root.chmod(0o700)
        cli._atomic_write_report(
            report_path, b'{"status":"safe"}', private_root=private_root,
        )
        lock = report_path.parent / ".report.lock"
        c.true(lock.is_file(), "the serialized publisher retains its lock file")
        c.eq(stat.S_IMODE(lock.stat().st_mode), 0o600,
             "the publication lock is owner-only")
        lock.chmod(0o644)
        c.raises(
            lambda: cli._atomic_write_report(
                report_path, b'{"status":"unsafe-lock"}',
                private_root=private_root,
            ),
            ValueError,
            "an unsafe existing publication lock fails closed",
        )


def test_private_publisher_stays_on_held_directory_after_parent_component_swap(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private_root = root / "benchmarks"
        report_path = private_root / "development" / "fixed" / "report.json"
        parked = private_root / "parked-development"
        original_token_hex = cli.secrets.token_hex

        def swap_parent(_length):
            development = private_root / "development"
            development.rename(parked)
            development.mkdir(mode=0o700)
            (development / "fixed").mkdir(mode=0o700)
            return "f" * 32

        cli.secrets.token_hex = swap_parent
        try:
            cli._atomic_write_report(
                report_path, b'{"status":"anchored"}',
                private_root=private_root,
            )
        finally:
            cli.secrets.token_hex = original_token_hex
        c.true(not report_path.exists(),
               "a swapped-in pathname tree receives no report")
        c.eq(
            (parked / "fixed" / "report.json").read_bytes(),
            b'{"status":"anchored"}',
            "publication remains anchored to the held directory inode",
        )


def test_private_publisher_serializes_two_writers_with_one_lock(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private_root = root / "benchmarks"
        report_path = private_root / "development" / "fixed" / "report.json"
        first_payload = b'{"writer":"first"}'
        second_payload = b'{"writer":"second"}'
        first_inside = threading.Event()
        release_first = threading.Event()
        second_inside = threading.Event()
        failures = []
        original_write = cli.os.write

        def guarded_write(descriptor, payload):
            if payload == first_payload and not first_inside.is_set():
                first_inside.set()
                if not release_first.wait(5.0):
                    raise RuntimeError("first writer was not released")
            elif payload == second_payload:
                second_inside.set()
            return original_write(descriptor, payload)

        def publish(payload):
            try:
                cli._atomic_write_report(
                    report_path, payload, private_root=private_root,
                )
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        cli.os.write = guarded_write
        first = threading.Thread(target=publish, args=(first_payload,))
        second = threading.Thread(target=publish, args=(second_payload,))
        try:
            first.start()
            c.true(first_inside.wait(5.0), "the first writer reaches its temp file")
            second.start()
            c.true(not second_inside.wait(0.25),
                   "the second writer blocks on the held publication lock")
            release_first.set()
            first.join(5.0)
            second.join(5.0)
        finally:
            release_first.set()
            first.join(5.0)
            second.join(5.0)
            cli.os.write = original_write
        c.eq(failures, [], "both serialized writers complete without error")
        c.true(not first.is_alive() and not second.is_alive(),
               "both writer threads terminate")
        c.eq(report_path.read_bytes(), second_payload,
             "the lock gives the later writer one complete atomic turn")


def test_authenticated_chain_is_exact_and_rejects_nonfusion_or_seed_drift(c: Check):
    cli = _load_cli()
    calls = []
    authorization = object()
    common = {"common_contract_sha256": "5" * 64}
    smoke = {"report_sha256": "a" * 64}
    selection = {"report_sha256": "b" * 64, "selected_arm": "fusion"}
    winner = {
        "report": {"selected_arm": "fusion", "seeds": [0, 1, 2]},
        "report_sha256": "6" * 64,
        "selected_arm": "fusion",
        "checkpoints": {seed: {} for seed in range(3)},
    }
    originals = {
        name: getattr(cli.pretrain_cli, name)
        for name in (
            "_focused_trainer_sha256", "_authorize_focused_bridge",
            "_focused_common_contract", "_validate_focused_smoke_phase",
            "_validate_focused_selection_phase", "_validate_focused_winner_phase",
        )
    }
    cli.pretrain_cli._focused_trainer_sha256 = lambda: "3" * 64
    cli.pretrain_cli._authorize_focused_bridge = lambda root, key, *, producer_sha256: (
        calls.append(("authorize", root, key, producer_sha256)) or authorization
    )
    cli.pretrain_cli._focused_common_contract = lambda auth: (
        calls.append(("common", auth)) or common
    )
    cli.pretrain_cli._validate_focused_smoke_phase = lambda directory, **kwargs: (
        calls.append(("smoke", directory, kwargs)) or smoke
    )
    cli.pretrain_cli._validate_focused_selection_phase = lambda directory, **kwargs: (
        calls.append(("selection", directory, kwargs)) or selection
    )
    cli.pretrain_cli._validate_focused_winner_phase = lambda directory, **kwargs: (
        calls.append(("winner", directory, kwargs)) or winner
    )
    try:
        result = cli._authenticate_winner_chain()
        c.eq(result, (authorization, common, winner, "3" * 64),
             "the existing authenticated chain is returned intact")
        c.eq([row[0] for row in calls],
             ["authorize", "common", "smoke", "selection", "winner"],
             "authorization proceeds only in smoke-selection-winner order")
        bad_arm = dict(winner, selected_arm="landmark_only")
        cli.pretrain_cli._validate_focused_winner_phase = lambda *args, **kwargs: bad_arm
        c.raises(cli._authenticate_winner_chain, ValueError,
                 "a non-Fusion authenticated winner fails closed")
        bad_seeds = dict(winner, report={"selected_arm": "fusion", "seeds": [0, 1]})
        cli.pretrain_cli._validate_focused_winner_phase = lambda *args, **kwargs: bad_seeds
        c.raises(cli._authenticate_winner_chain, ValueError,
                 "authenticated report seed drift fails closed")
    finally:
        for name, value in originals.items():
            setattr(cli.pretrain_cli, name, value)


def _synthetic_authenticated_lineage(*, receipt_suffix: str = "a"):
    authorization = SimpleNamespace(bridge_generation_sha256="4" * 64)
    common = {"common_contract_sha256": "5" * 64}
    winner = {
        "report_sha256": "6" * 64,
        "checkpoints": {
            seed: {
                "checkpoint_fingerprint": str(7 + seed) * 64,
                "receipt_file_sha256": chr(ord(receipt_suffix) + seed) * 64,
            }
            for seed in range(3)
        },
    }
    return authorization, common, winner, "3" * 64


def _install_synthetic_benchmark(cli, chains, source_hashes):
    chain_values = iter(chains)
    digest_values = iter(source_hashes)
    originals = {
        "authenticate": cli._authenticate_winner_chain,
        "heldout": cli._heldout_inputs,
        "models": cli._models_and_clean_metrics,
        "safe_source": cli._safe_source_sha256,
        "evaluate": cli.fusion_core.evaluate_fusion_conditions,
        "aggregate": cli.fusion_core.aggregate_condition_metrics,
        "authority": cli.pretrain_cli._require_focused_authority_unchanged,
    }
    cli._authenticate_winner_chain = lambda: next(chain_values)
    cli._heldout_inputs = lambda _authorization, _common: {}
    cli._models_and_clean_metrics = lambda _winner: ({}, {}, {})
    cli._safe_source_sha256 = lambda _path, _label: next(digest_values)
    cli.fusion_core.evaluate_fusion_conditions = lambda **_kwargs: (
        _all_condition_rows()
    )
    cli.fusion_core.aggregate_condition_metrics = aggregate_condition_metrics
    cli.pretrain_cli._require_focused_authority_unchanged = lambda authorization: (
        authorization
    )
    return originals


def _restore_synthetic_benchmark(cli, originals):
    cli._authenticate_winner_chain = originals["authenticate"]
    cli._heldout_inputs = originals["heldout"]
    cli._models_and_clean_metrics = originals["models"]
    cli._safe_source_sha256 = originals["safe_source"]
    cli.fusion_core.evaluate_fusion_conditions = originals["evaluate"]
    cli.fusion_core.aggregate_condition_metrics = originals["aggregate"]
    cli.pretrain_cli._require_focused_authority_unchanged = originals["authority"]


def test_benchmark_rejects_source_digest_drift_across_evaluation(c: Check):
    cli = _load_cli()
    chain = _synthetic_authenticated_lineage()
    originals = _install_synthetic_benchmark(
        cli, [chain, chain], ["1" * 64, "2" * 64, "9" * 64, "2" * 64],
    )
    try:
        c.raises(cli.run_benchmark, ValueError,
                 "source drift during long evaluation fails closed")
    finally:
        _restore_synthetic_benchmark(cli, originals)


def test_benchmark_rejects_authenticated_lineage_drift_across_evaluation(c: Check):
    cli = _load_cli()
    before = _synthetic_authenticated_lineage()
    after = _synthetic_authenticated_lineage(receipt_suffix="d")
    originals = _install_synthetic_benchmark(
        cli, [before, after], ["1" * 64, "2" * 64, "1" * 64, "2" * 64],
    )
    try:
        c.raises(cli.run_benchmark, ValueError,
                 "checkpoint receipt lineage drift fails closed")
    finally:
        _restore_synthetic_benchmark(cli, originals)


def test_mocked_main_validates_and_publishes_the_fixed_report(c: Check):
    cli = _load_cli()
    report = _valid_public_report(cli)
    calls = []
    original_edge = getattr(cli, "_require_publication_edge", None)
    originals = (cli.run_benchmark, cli._atomic_write_report, cli.DEFAULT_REPORT_PATH)
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "benchmarks" / "development" / "fixed" / "report.json"
        cli.DEFAULT_REPORT_PATH = target
        cli.run_benchmark = lambda: report
        cli._require_publication_edge = lambda observed: calls.append(
            ("edge", observed)
        )
        cli._atomic_write_report = lambda path, payload, *, private_root: calls.append(
            ("write", path, payload, private_root)
        )
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                cli.main([])
        finally:
            cli.run_benchmark, cli._atomic_write_report, cli.DEFAULT_REPORT_PATH = originals
            if original_edge is None:
                del cli._require_publication_edge
            else:
                cli._require_publication_edge = original_edge
    expected_payload = cli.canonical_json_bytes(cli.validate_public_report(report))
    c.eq(calls, [
        ("edge", report),
        ("write", target, expected_payload, target.parents[2]),
    ], "main reauthorizes the publication edge before its only write")
    line = output.getvalue().strip()
    c.true(line.startswith("status=complete report_sha256=") and len(line) == 94,
           "stdout contains only status and the report digest")
    c.true("/" not in line and "\\" not in line,
           "stdout never reveals a path")


def test_real_publication_edge_rejects_source_and_lineage_drift_before_write(c: Check):
    cli = _load_cli()
    cases = []
    source_report = _valid_public_report(cli)
    cases.append((
        source_report,
        ["9" * 64, "2" * 64],
        _synthetic_authenticated_lineage(),
    ))
    lineage_report = _valid_public_report(cli)
    changed_lineage = list(_synthetic_authenticated_lineage())
    changed_winner = dict(changed_lineage[2])
    changed_winner["report_sha256"] = "9" * 64
    changed_lineage[2] = changed_winner
    cases.append((lineage_report, ["1" * 64, "2" * 64], tuple(changed_lineage)))

    for report, source_hashes, chain in cases:
        writes = []
        digests = iter(source_hashes)
        originals = (
            cli.run_benchmark,
            cli._atomic_write_report,
            cli._safe_source_sha256,
            cli._authenticate_winner_chain,
        )
        cli.run_benchmark = lambda report=report: report
        cli._atomic_write_report = lambda *args, **kwargs: writes.append(True)
        cli._safe_source_sha256 = lambda _path, _label: next(digests)
        cli._authenticate_winner_chain = lambda chain=chain: chain
        try:
            c.raises(lambda: cli.main([]), ValueError,
                     "publication-edge drift fails before serialization")
        finally:
            (
                cli.run_benchmark,
                cli._atomic_write_report,
                cli._safe_source_sha256,
                cli._authenticate_winner_chain,
            ) = originals
        c.eq(writes, [], "no report write occurs after publication-edge drift")


if __name__ == "__main__":
    run_all("test_focused_fusion_robustness", dict(globals()))
