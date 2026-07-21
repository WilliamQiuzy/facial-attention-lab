"""Pure protocols and perturbations for the focused Fusion benchmark.

This module does not read benchmark data, checkpoint state, or output
artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    ROUND_HALF_EVEN,
    localcontext,
)
import math
from typing import Any, Final, Optional

import torch

from ..models.dynamic_landmark import (
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
)
from ..pretraining import dynamic_landmark_ssl as ssl_core


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
_MAX_METRIC: Final[Decimal] = Decimal("1000000000")
_MAX_METRIC_DIGITS: Final[int] = 64
_MIN_METRIC_EXPONENT: Final[int] = -100
_MAX_METRIC_EXPONENT: Final[int] = 100
_METRIC_CONTEXT: Final[Context] = Context(
    prec=32,
    rounding=ROUND_HALF_EVEN,
    Emin=-100,
    Emax=100,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def _exact_dict(value: object, keys: tuple[str, ...], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact schema dictionary")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must have exact type str")
    if set(value) != set(keys) or len(value) != len(keys):
        raise ValueError(f"{name} must be an exact schema dictionary")
    return value


def canonical_metric(value: object) -> Decimal:
    """Return one finite, nonnegative metric rounded to the protocol quantum."""
    if type(value) not in (int, float, Decimal):
        raise TypeError("metric must have exact numeric scalar type")
    try:
        if type(value) is int:
            if value < 0 or value > 1_000_000_000:
                raise ValueError("metric must be in the closed interval [0, 1e9]")
            metric = Decimal(value)
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("metric must be finite")
            if value < 0 or value > 1_000_000_000:
                raise ValueError("metric must be in the closed interval [0, 1e9]")
            metric = Decimal(str(value))
        else:
            metric = value
            if not metric.is_finite():
                raise ValueError("metric must be finite")

        representation = metric.as_tuple()
        if (
            len(representation.digits) > _MAX_METRIC_DIGITS
            or representation.exponent < _MIN_METRIC_EXPONENT
            or representation.exponent > _MAX_METRIC_EXPONENT
        ):
            raise ValueError("metric representation exceeds protocol limits")
        if metric < 0 or metric > _MAX_METRIC:
            raise ValueError("metric must be in the closed interval [0, 1e9]")
        with localcontext(_METRIC_CONTEXT) as context:
            result = context.quantize(metric, _METRIC_QUANTUM)
            return result.copy_abs() if result.is_zero() else result
    except (DecimalException, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("metric is invalid under the fixed protocol context") from exc


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


def _validate_public_metric_bundle(value: object) -> dict[str, dict[str, object]]:
    bundle = _exact_dict(value, _BASELINES, "public metric bundle")
    for baseline in _BASELINES:
        source = _exact_dict(bundle[baseline], _METRIC_KEYS, "public metric baseline")
        raw = _exact_dict(source["raw_mae"], _RAW_MAE_KEYS, "public raw_mae")
        values = [raw[key] for key in _RAW_MAE_KEYS]
        values.extend(source[key] for key in _METRIC_KEYS[1:])
        if any(type(metric) is not float for metric in values):
            raise ValueError("public metric values must have exact type float")
    return validate_metric_bundle(bundle)


def require_clean_replay(observed_rows: object, expected_by_seed: object) -> None:
    """Fail closed unless the three clean replay metrics exactly agree."""
    if type(observed_rows) is not list or len(observed_rows) != 3:
        raise ValueError("clean replay requires exactly three observed rows")
    if (
        type(expected_by_seed) is not dict
        or len(expected_by_seed) != 3
    ):
        raise ValueError("clean replay requires exact expected seeds 0, 1, 2")
    if any(type(seed) is not int for seed in expected_by_seed):
        raise ValueError("clean replay expected seed keys require exact type int")
    if any(seed < 0 or seed > 2 for seed in expected_by_seed):
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
    try:
        with localcontext(_METRIC_CONTEXT):
            return sum(values, Decimal(0)) / len(values)
    except DecimalException as exc:
        raise ValueError("metric mean is invalid") from exc


def _decimal_sample_sd(values: list[Decimal], mean: Decimal) -> Decimal:
    try:
        with localcontext(_METRIC_CONTEXT):
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            return variance.sqrt()
    except DecimalException as exc:
        raise ValueError("metric sample deviation is invalid") from exc


def _json_finite_decimal(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("aggregate must be representable as a finite JSON number")
    return 0.0 if result == 0.0 else result


def _decimal_degradation(mean: Decimal, clean_mean: Decimal) -> Decimal:
    try:
        with localcontext(_METRIC_CONTEXT):
            return Decimal(100) * (mean / clean_mean - Decimal(1))
    except DecimalException as exc:
        raise ValueError("condition degradation is invalid") from exc


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
        degradation = (
            Decimal(0) if name == "clean_fusion"
            else _decimal_degradation(mean, clean_mean)
        )
        if (
            not degradation.is_finite()
            or degradation < Decimal(-100)
            or degradation > Decimal("1e16")
        ):
            raise ValueError("condition degradation is outside the publication range")
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
    """Reconstruct only the exact deidentified aggregate publication schema."""
    payload = _exact_dict(value, ("conditions",), "aggregate payload")
    conditions = payload["conditions"]
    if type(conditions) is not list or len(conditions) != len(BENCHMARK_CONDITIONS):
        raise ValueError("aggregate payload requires ten ordered conditions")

    normalized_conditions: list[dict[str, object]] = []
    for index, condition_spec in enumerate(BENCHMARK_CONDITIONS):
        entry = _exact_dict(
            conditions[index],
            ("condition", "seed_rows", "aggregates", "degradation_percent_vs_clean"),
            "aggregate condition",
        )
        if type(entry["condition"]) is not str or entry["condition"] != condition_spec.name:
            raise ValueError("aggregate conditions must use registered order")
        seed_rows = entry["seed_rows"]
        if type(seed_rows) is not list or len(seed_rows) != 3:
            raise ValueError("aggregate condition requires three ordered seed rows")
        normalized_rows: list[dict[str, object]] = []
        for seed in range(3):
            row = _exact_dict(
                seed_rows[seed], ("condition", "seed", "metrics"), "aggregate seed row"
            )
            if type(row["condition"]) is not str or row["condition"] != condition_spec.name:
                raise ValueError("aggregate seed condition is invalid")
            if type(row["seed"]) is not int or row["seed"] != seed:
                raise ValueError("aggregate seed rows must be ordered 0, 1, 2")
            normalized_rows.append({
                "condition": condition_spec.name,
                "seed": seed,
                "metrics": _validate_public_metric_bundle(row["metrics"]),
            })

        aggregates = _validate_aggregate_bundle(entry["aggregates"])
        degradation = entry["degradation_percent_vs_clean"]
        if (
            type(degradation) is not float
            or not math.isfinite(degradation)
            or degradation < -100.0
            or degradation > 1e16
        ):
            raise ValueError("aggregate degradation is outside the publication range")
        normalized_conditions.append({
            "condition": condition_spec.name,
            "seed_rows": normalized_rows,
            "aggregates": aggregates,
            "degradation_percent_vs_clean": (
                0.0 if degradation == 0.0 else degradation
            ),
        })
    return {"conditions": normalized_conditions}


def _validate_aggregate_bundle(value: object) -> dict[str, dict[str, object]]:
    bundle = _exact_dict(value, _BASELINES, "aggregate bundle")
    normalized: dict[str, dict[str, object]] = {}
    for baseline in _BASELINES:
        source = _exact_dict(bundle[baseline], _METRIC_KEYS, "aggregate baseline")
        raw = _exact_dict(source["raw_mae"], _RAW_MAE_KEYS, "aggregate raw_mae")
        normalized[baseline] = {
            "raw_mae": {
                key: _validate_metric_summary(raw[key]) for key in _RAW_MAE_KEYS
            },
            "standardized_mae": _validate_metric_summary(source["standardized_mae"]),
            "standardized_smooth_l1": _validate_metric_summary(
                source["standardized_smooth_l1"]
            ),
        }
    return normalized


def _validate_metric_summary(value: object) -> dict[str, float]:
    summary = _exact_dict(value, ("mean", "sample_sd"), "metric summary")
    result: dict[str, float] = {}
    for key in ("mean", "sample_sd"):
        metric = summary[key]
        if (
            type(metric) is not float
            or not math.isfinite(metric)
            or metric < 0
            or metric > 1_000_000_000.0
        ):
            raise ValueError("aggregate metric summary must contain finite floats")
        result[key] = 0.0 if metric == 0.0 else metric
    return result


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


def _validate_model_map(value: object, name: str) -> dict[int, ssl_core.DynamicLandmarkSSLModel]:
    """Validate one closed seed-to-real-model map without hostile comparisons."""
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact dictionary")
    if any(type(key) is not int for key in value):
        raise ValueError(f"{name} seed keys must have exact type int")
    if tuple(sorted(value)) != (0, 1, 2):
        raise ValueError(f"{name} must contain exactly seeds 0, 1, and 2")
    for seed in range(3):
        model = value[seed]
        if type(model) is not ssl_core.DynamicLandmarkSSLModel:
            raise ValueError(f"{name} values must be exact DynamicLandmarkSSLModel instances")
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} model state must be finite and CPU-resident")
    return value


def _validate_temporal_provenance(
    timestamps: object,
    source_frame_indices: object,
    leading_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(timestamps, torch.Tensor)
        or timestamps.device.type != "cpu"
        or timestamps.layout != torch.strided
        or tuple(timestamps.shape) != leading_shape
        or not timestamps.is_floating_point()
        or not bool(torch.isfinite(timestamps).all())
    ):
        raise ValueError("timestamps must be finite CPU floating tensors matching features")
    if (
        not isinstance(source_frame_indices, torch.Tensor)
        or source_frame_indices.device.type != "cpu"
        or source_frame_indices.layout != torch.strided
        or tuple(source_frame_indices.shape) != leading_shape
        or source_frame_indices.dtype != torch.int64
    ):
        raise ValueError("source frame indices must be CPU int64 tensors matching features")
    return timestamps, source_frame_indices


def _condition_metrics(
    *,
    trained: ssl_core.DynamicLandmarkSSLModel,
    fresh: ssl_core.DynamicLandmarkSSLModel,
    model_features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    model_reconstruction_mask: torch.Tensor,
    target: torch.Tensor,
    scoring_reconstruction_mask: torch.Tensor,
    scaler: ssl_core.SourceScaler,
    split: ssl_core.SSLGroupSplit,
    evaluated_indices: object,
    group_ids: object,
    input_arm: str,
) -> dict[str, dict[str, object]]:
    trained_prediction = trained(
        model_features, valid_mask, timestamps, source_frame_indices,
        reconstruction_mask=model_reconstruction_mask, source="mayo", input_arm=input_arm,
    )
    fresh_prediction = fresh(
        model_features, valid_mask, timestamps, source_frame_indices,
        reconstruction_mask=model_reconstruction_mask, source="mayo", input_arm=input_arm,
    )
    report = ssl_core.reconstruction_report(
        trained_prediction, fresh_prediction, target, scoring_reconstruction_mask,
        baseline=scaler, split=split, evaluated_indices=evaluated_indices,
        group_ids=group_ids, source=ssl_core.MAYO_SOURCE,
    )
    common = report["common_target_metrics"]
    if type(common) is not dict:
        raise ValueError("reconstruction report common target metrics are malformed")
    return validate_metric_bundle({
        "trained": common["trained"],
        "fresh_untrained": common["untrained"],
        "train_mean": common["train_mean"],
    })


def evaluate_fusion_conditions(
    *,
    trained_models: object,
    fresh_models: object,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: object,
    source_frame_indices: object,
    target_mask: torch.Tensor,
    scaler: object,
    split: object,
    evaluated_indices: object,
    group_ids: object,
    expected_clean_metrics_by_seed: object,
) -> list[dict[str, object]]:
    """Evaluate the closed Fusion robustness grid with clean replay gated first.

    Models run under ``eval`` and ``no_grad``; their caller-visible training modes
    are restored even when the clean replay gate rejects the result.
    """
    trained = _validate_model_map(trained_models, "trained_models")
    fresh = _validate_model_map(fresh_models, "fresh_models")
    if any(trained[seed] is fresh[other] for seed in range(3) for other in range(3)):
        raise ValueError("fresh models must be separate objects from trained models")
    if type(scaler) is not ssl_core.SourceScaler:
        raise ValueError("scaler must have exact type SourceScaler")
    if type(split) is not ssl_core.SSLGroupSplit:
        raise ValueError("split must have exact type SSLGroupSplit")

    clean_features, clean_model_mask, clean_arm = build_condition_inputs(
        features, valid_mask, target_mask, BENCHMARK_CONDITIONS[0],
    )
    checked_timestamps, checked_source_indices = _validate_temporal_provenance(
        timestamps, source_frame_indices, tuple(features.shape[:-1]),
    )
    if not torch.equal(clean_model_mask, target_mask) or clean_arm != ARM_FUSION:
        raise RuntimeError("clean Fusion registry contradicts the benchmark protocol")

    models = tuple((*trained.values(), *fresh.values()))
    modes = tuple(model.training for model in models)
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            rows: list[dict[str, object]] = []
            for seed in range(3):
                rows.append({
                    "condition": "clean_fusion",
                    "seed": seed,
                    "metrics": _condition_metrics(
                        trained=trained[seed], fresh=fresh[seed],
                        model_features=clean_features, valid_mask=valid_mask,
                        timestamps=checked_timestamps,
                        source_frame_indices=checked_source_indices,
                        model_reconstruction_mask=target_mask, target=features,
                        scoring_reconstruction_mask=target_mask, scaler=scaler,
                        split=split, evaluated_indices=evaluated_indices,
                        group_ids=group_ids, input_arm=ARM_FUSION,
                    ),
                })
            require_clean_replay(rows, expected_clean_metrics_by_seed)

            for condition in BENCHMARK_CONDITIONS[1:]:
                model_features, model_mask, input_arm = build_condition_inputs(
                    features, valid_mask, target_mask, condition,
                )
                for seed in range(3):
                    rows.append({
                        "condition": condition.name,
                        "seed": seed,
                        "metrics": _condition_metrics(
                            trained=trained[seed], fresh=fresh[seed],
                            model_features=model_features, valid_mask=valid_mask,
                            timestamps=checked_timestamps,
                            source_frame_indices=checked_source_indices,
                            model_reconstruction_mask=model_mask, target=features,
                            scoring_reconstruction_mask=target_mask, scaler=scaler,
                            split=split, evaluated_indices=evaluated_indices,
                            group_ids=group_ids, input_arm=input_arm,
                        ),
                    })
    finally:
        for model, mode in zip(models, modes):
            model.train(mode)
    return rows
