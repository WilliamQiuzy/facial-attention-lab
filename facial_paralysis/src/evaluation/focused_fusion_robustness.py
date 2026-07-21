"""Pure protocols and perturbations for the focused Fusion benchmark.

This module does not read benchmark data, checkpoint state, or output
artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import math
from typing import Any, Final, Optional

import torch

from ..models.dynamic_landmark import (
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
)


_ConditionSpec = tuple[str, str, Optional[float], Optional[float], Optional[int]]

_FROZEN_BENCHMARK_SPECS: Final[tuple[_ConditionSpec, ...]] = (
    ("clean_fusion", ARM_FUSION, None, None, None),
    ("mask_landmarks", ARM_BLENDSHAPE, None, None, None),
    ("mask_blendshapes", ARM_LANDMARK, None, None, None),
    ("context_dropout_10pct", ARM_FUSION, 0.10, None, 41010),
    ("context_dropout_25pct", ARM_FUSION, 0.25, None, 41025),
    ("context_dropout_50pct", ARM_FUSION, 0.50, None, 41050),
    ("landmark_noise_0.10sd", ARM_FUSION, None, 0.10, 52010),
    ("landmark_noise_0.25sd", ARM_FUSION, None, 0.25, 52025),
    ("landmark_noise_0.50sd", ARM_FUSION, None, 0.50, 52050),
    ("frame_order_shuffle", ARM_FUSION, None, None, 63000),
)


@dataclass(frozen=True, init=False)
class BenchmarkCondition:
    """One member of the closed focused-Fusion benchmark protocol."""

    __slots__ = (
        "name",
        "input_arm",
        "context_dropout_probability",
        "landmark_noise_sd",
        "rng_seed",
    )

    name: str
    input_arm: str
    context_dropout_probability: Optional[float]
    landmark_noise_sd: Optional[float]
    rng_seed: Optional[int]

    def __init__(
        self,
        name: str,
        input_arm: str,
        context_dropout_probability: Optional[float] = None,
        landmark_noise_sd: Optional[float] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "input_arm", input_arm)
        object.__setattr__(
            self, "context_dropout_probability", context_dropout_probability
        )
        object.__setattr__(self, "landmark_noise_sd", landmark_noise_sd)
        object.__setattr__(self, "rng_seed", rng_seed)
        self.__post_init__()

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("BenchmarkCondition is closed and cannot be subclassed")

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("name must have exact type str")
        if type(self.input_arm) is not str:
            raise TypeError("input_arm must have exact type str")
        if (
            self.context_dropout_probability is not None
            and type(self.context_dropout_probability) is not float
        ):
            raise TypeError(
                "context_dropout_probability must have exact type float or be None"
            )
        if (
            self.landmark_noise_sd is not None
            and type(self.landmark_noise_sd) is not float
        ):
            raise TypeError(
                "landmark_noise_sd must have exact type float or be None"
            )
        if self.rng_seed is not None and type(self.rng_seed) is not int:
            raise TypeError("rng_seed must have exact type int or be None")
        spec = (
            self.name,
            self.input_arm,
            self.context_dropout_probability,
            self.landmark_noise_sd,
            self.rng_seed,
        )
        if spec not in _FROZEN_BENCHMARK_SPECS:
            raise ValueError(
                "benchmark conditions must match the frozen protocol registry"
            )


BENCHMARK_CONDITIONS: Final[tuple[BenchmarkCondition, ...]] = tuple(
    BenchmarkCondition(*spec) for spec in _FROZEN_BENCHMARK_SPECS
)


_BASELINES: Final[tuple[str, ...]] = (
    "trained", "fresh_untrained", "train_mean",
)
_METRIC_KEYS: Final[tuple[str, ...]] = (
    "raw_mae", "standardized_mae", "standardized_smooth_l1",
)
_RAW_MAE_KEYS: Final[tuple[str, ...]] = (
    "blendshape72", "clinical23", "equal_block_macro", "full95",
)
_METRIC_QUANTUM: Final[Decimal] = Decimal("0.00001")


def _exact_dict(value: object, keys: tuple[str, ...], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys) or len(value) != len(keys):
        raise ValueError(f"{name} must be an exact schema dictionary")
    return value


def canonical_metric(value: object) -> Decimal:
    """Return one finite, nonnegative metric rounded to the protocol quantum."""
    if type(value) not in (int, float, Decimal):
        raise TypeError("metric must have exact numeric scalar type")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("metric must be finite")
    try:
        metric = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("metric must be finite") from exc
    if not metric.is_finite():
        raise ValueError("metric must be finite")
    if metric < 0:
        raise ValueError("metric must be nonnegative")
    precision = max(50, abs(metric.adjusted()) + 20, len(metric.as_tuple().digits) + 10)
    try:
        with localcontext() as context:
            context.prec = precision
            return metric.quantize(_METRIC_QUANTUM, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise ValueError("metric cannot be represented at protocol precision") from exc


def _json_metric(value: object) -> float:
    result = float(canonical_metric(value))
    if not math.isfinite(result):
        raise ValueError("metric must be representable as a finite JSON number")
    return result


def _canonical_metric_bundle(value: object) -> dict[str, dict[str, object]]:
    """Validate the closed metric schema while retaining exact Decimal values."""
    bundle = _exact_dict(value, _BASELINES, "metric bundle")
    normalized: dict[str, dict[str, object]] = {}
    for baseline in _BASELINES:
        source = _exact_dict(bundle[baseline], _METRIC_KEYS, baseline)
        raw = _exact_dict(source["raw_mae"], _RAW_MAE_KEYS, "raw_mae")
        normalized[baseline] = {
            "raw_mae": {
                key: canonical_metric(raw[key]) for key in _RAW_MAE_KEYS
            },
            "standardized_mae": canonical_metric(source["standardized_mae"]),
            "standardized_smooth_l1": canonical_metric(source["standardized_smooth_l1"]),
        }
    return normalized


def _json_metric_bundle(value: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        baseline: {
            "raw_mae": {
                key: _json_metric(value[baseline]["raw_mae"][key])
                for key in _RAW_MAE_KEYS
            },
            "standardized_mae": _json_metric(value[baseline]["standardized_mae"]),
            "standardized_smooth_l1": _json_metric(
                value[baseline]["standardized_smooth_l1"]
            ),
        }
        for baseline in _BASELINES
    }


def validate_metric_bundle(value: object) -> dict[str, dict[str, object]]:
    """Validate the closed metric schema and return JSON-safe canonical values."""
    return _json_metric_bundle(_canonical_metric_bundle(value))


def require_clean_replay(observed_rows: object, expected_by_seed: object) -> None:
    """Fail closed unless the three clean replay metrics exactly agree."""
    if type(observed_rows) is not list or len(observed_rows) != 3:
        raise ValueError("clean replay requires exactly three observed rows")
    if (
        type(expected_by_seed) is not dict
        or set(expected_by_seed) != {0, 1, 2}
        or any(type(seed) is not int for seed in expected_by_seed)
    ):
        raise ValueError("clean replay requires exact expected seeds 0, 1, 2")
    seen: set[int] = set()
    for row in observed_rows:
        item = _exact_dict(row, ("condition", "seed", "metrics"), "clean replay row")
        if type(item["condition"]) is not str or item["condition"] != "clean_fusion":
            raise ValueError("clean replay rows must be clean_fusion")
        if type(item["seed"]) is not int or item["seed"] not in {0, 1, 2}:
            raise ValueError("clean replay seed is invalid")
        if item["seed"] in seen:
            raise ValueError("clean replay seeds must be unique")
        seen.add(item["seed"])
        observed = _canonical_metric_bundle(item["metrics"])
        expected = _canonical_metric_bundle(expected_by_seed[item["seed"]])
        if observed != expected:
            raise ValueError("clean replay metrics do not match expected values")
    if seen != {0, 1, 2}:
        raise ValueError("clean replay seed set is incomplete")


def _decimal_mean(values: list[Decimal]) -> Decimal:
    precision = max(50, max(abs(value.adjusted()) for value in values) + 30)
    with localcontext() as context:
        context.prec = precision
        return sum(values, Decimal(0)) / len(values)


def _decimal_sample_sd(values: list[Decimal], mean: Decimal) -> Decimal:
    precision = max(50, max(abs(value.adjusted()) for value in values) + 30)
    with localcontext() as context:
        context.prec = precision
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return variance.sqrt()


def _json_finite_decimal(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("aggregate must be representable as a finite JSON number")
    return result


def _summary(values: list[Decimal]) -> dict[str, float]:
    mean = _decimal_mean(values)
    return {
        "mean": _json_finite_decimal(mean),
        "sample_sd": _json_finite_decimal(_decimal_sample_sd(values, mean)),
    }


def _aggregate_bundle(bundles: list[dict[str, dict[str, object]]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for baseline in _BASELINES:
        raw_result: dict[str, object] = {}
        for key in _RAW_MAE_KEYS:
            values = [bundle[baseline]["raw_mae"][key] for bundle in bundles]
            raw_result[key] = _summary(values)
        result[baseline] = {
            "raw_mae": raw_result,
            **{
                key: _summary([bundle[baseline][key] for bundle in bundles])
                for key in _METRIC_KEYS[1:]
            },
        }
    return result


def aggregate_condition_metrics(rows: object) -> dict[str, object]:
    """Aggregate the complete closed condition-by-seed grid without identities."""
    if type(rows) is not list or len(rows) != 30:
        raise ValueError("condition metrics require exactly thirty rows")
    registered = tuple(condition.name for condition in BENCHMARK_CONDITIONS)
    grouped: dict[str, dict[int, dict[str, object]]] = {
        name: {} for name in registered
    }
    for row in rows:
        item = _exact_dict(row, ("condition", "seed", "metrics"), "condition row")
        condition, seed = item["condition"], item["seed"]
        if type(condition) is not str or condition not in grouped:
            raise ValueError("condition must be registered")
        if type(seed) is not int or seed not in {0, 1, 2}:
            raise ValueError("seed must be exactly 0, 1, or 2")
        if seed in grouped[condition]:
            raise ValueError("condition seed rows must be unique")
        grouped[condition][seed] = _canonical_metric_bundle(item["metrics"])
    if any(set(by_seed) != {0, 1, 2} for by_seed in grouped.values()):
        raise ValueError("condition seed grid is incomplete")

    aggregates = {
        name: _aggregate_bundle([grouped[name][seed] for seed in range(3)])
        for name in registered
    }
    clean_mean = _decimal_mean([
        grouped["clean_fusion"][seed]["trained"]["raw_mae"]["equal_block_macro"]
        for seed in range(3)
    ])
    if not clean_mean.is_finite() or clean_mean == 0:
        raise ValueError("clean trained macro mean must be finite and nonzero")
    conditions: list[dict[str, object]] = []
    for name in registered:
        mean = _decimal_mean([
            grouped[name][seed]["trained"]["raw_mae"]["equal_block_macro"]
            for seed in range(3)
        ])
        if not mean.is_finite():
            raise ValueError("condition trained macro mean must be finite")
        degradation = Decimal(0) if name == "clean_fusion" else Decimal(100) * (
            mean / clean_mean - Decimal(1)
        )
        if not degradation.is_finite():
            raise ValueError("condition degradation must be finite")
        conditions.append({
            "condition": name,
            "seed_rows": [
                {
                    "condition": name,
                    "seed": seed,
                    "metrics": _json_metric_bundle(grouped[name][seed]),
                }
                for seed in range(3)
            ],
            "aggregates": aggregates[name],
            "degradation_percent_vs_clean": _json_finite_decimal(degradation),
        })
    return {"conditions": conditions}


def validate_deidentified_payload(value: object) -> object:
    """Return a fresh JSON-safe payload, rejecting identifiers and path material."""
    forbidden = (
        "path", "recording_id", "group_id", "sample_id", "source_unit",
        "private_key", "authority_hmac",
    )

    def validate(item: object) -> object:
        if type(item) is dict:
            result: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str or any(word in key.lower() for word in forbidden):
                    raise ValueError("payload contains a forbidden key")
                result[key] = validate(child)
            return result
        if type(item) is list:
            return [validate(child) for child in item]
        if type(item) is str:
            lower = item.lower()
            if (
                "/" in item
                or "\\" in item
                or item.startswith("~")
                or item in (".", "..")
                or (len(item) == 2 and item[0].isalpha() and item[1] == ":")
                or lower.startswith(
                ("rec_", "grp_", "sample_", "source_unit_")
                )
            ):
                raise ValueError("payload contains identifying or path-like string")
            return item
        if type(item) in (type(None), bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise ValueError("payload must be JSON-safe with finite numeric values")

    return validate(value)


def build_condition_inputs(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    target_mask: torch.Tensor,
    condition: BenchmarkCondition,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Build deterministic model inputs for one registered condition."""
    if type(condition) is not BenchmarkCondition:
        raise ValueError("condition must have exact type BenchmarkCondition")
    condition.__post_init__()
    if (
        not isinstance(features, torch.Tensor)
        or features.device.type != "cpu"
        or features.layout != torch.strided
        or features.ndim != 4
        or features.shape[0] < 1
        or tuple(features.shape[1:]) != (4, 32, 95)
        or features.dtype != torch.float32
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError(
            "features must be finite CPU float32 with shape (N, 4, 32, 95)"
        )
    for name, mask in (("valid_mask", valid_mask), ("target_mask", target_mask)):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.device.type != "cpu"
            or mask.layout != torch.strided
            or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(features.shape[:-1])
        ):
            raise ValueError(
                f"{name} must be a CPU bool tensor matching the feature leading shape"
            )
    if bool((target_mask & ~valid_mask).any()):
        raise ValueError("target_mask must be a subset of valid_mask")
    if bool((~target_mask.reshape(features.shape[0], -1).any(dim=1)).any()):
        raise ValueError("every sample must contain at least one target position")

    observed_context = valid_mask & ~target_mask
    if bool((~observed_context.reshape(features.shape[0], -1).any(dim=1)).any()):
        raise ValueError("every sample must contain observed context")

    model_features = features.clone()
    reconstruction_mask = target_mask.clone()
    if condition.context_dropout_probability is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(condition.rng_seed)
        random_draw = torch.rand(
            valid_mask.shape,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        drop = observed_context & (
            random_draw < condition.context_dropout_probability
        )
        reconstruction_mask = target_mask | drop
        remaining_context = valid_mask & ~reconstruction_mask
        if bool((~remaining_context.reshape(features.shape[0], -1).any(dim=1)).any()):
            raise ValueError("context dropout removed all context from a sample")
    elif condition.landmark_noise_sd is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(condition.rng_seed)
        noise = torch.randn(
            model_features[..., 72:95].shape,
            dtype=model_features.dtype,
            device="cpu",
            generator=generator,
        ).mul(condition.landmark_noise_sd)
        landmarks = model_features[..., 72:95]
        landmarks[observed_context] += noise[observed_context]
    elif condition.name == "frame_order_shuffle":
        generator = torch.Generator(device="cpu")
        assert condition.rng_seed is not None
        generator.manual_seed(condition.rng_seed)
        for sample in range(features.shape[0]):
            for window in range(features.shape[1]):
                context_indices = observed_context[sample, window].nonzero().flatten()
                permutation = torch.randperm(
                    context_indices.numel(), generator=generator,
                )
                model_features[sample, window, context_indices] = features[
                    sample, window, context_indices[permutation]
                ]

    return model_features, reconstruction_mask, condition.input_arm
