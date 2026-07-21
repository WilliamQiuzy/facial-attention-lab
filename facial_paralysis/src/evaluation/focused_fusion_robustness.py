"""Pure protocols and perturbations for the focused Fusion benchmark.

This module does not read benchmark data, checkpoint state, or output
artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

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
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("features must be finite CPU floats with shape (N, 4, 32, 95)")
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
        generator.manual_seed(63000)
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
