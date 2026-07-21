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


@dataclass(frozen=True)
class BenchmarkCondition:
    """One member of the closed focused-Fusion benchmark protocol."""

    name: str
    input_arm: str
    context_dropout_probability: Optional[float] = None
    landmark_noise_sd: Optional[float] = None
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
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
