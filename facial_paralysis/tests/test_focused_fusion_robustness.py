"""Protocol-contract tests for the focused Fusion robustness benchmark."""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.focused_fusion_robustness import (  # noqa: E402
    BENCHMARK_CONDITIONS,
    BenchmarkCondition,
)
from src.models.dynamic_landmark import (  # noqa: E402
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
)


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
            ARM_FUSION,
            ARM_BLENDSHAPE,
            ARM_LANDMARK,
            ARM_FUSION,
            ARM_FUSION,
            ARM_FUSION,
            ARM_FUSION,
            ARM_FUSION,
            ARM_FUSION,
            ARM_FUSION,
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


def test_caller_defined_protocols_fail(c: Check):
    clean = BENCHMARK_CONDITIONS[0]
    c.raises(lambda: BenchmarkCondition(
        name="caller_defined",
        input_arm=ARM_FUSION,
    ), ValueError, "callers cannot add arbitrary condition names")
    c.raises(lambda: replace(clean, context_dropout_probability=0.25),
             ValueError, "callers cannot alter a registered condition spec")


if __name__ == "__main__":
    run_all("test_focused_fusion_robustness", dict(globals()))
