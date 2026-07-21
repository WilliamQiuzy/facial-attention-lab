"""Frozen protocols for the focused Fusion robustness benchmark.

This module defines protocol metadata only.  It does not read benchmark data,
checkpoint state, or output artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

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
