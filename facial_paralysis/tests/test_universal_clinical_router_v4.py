from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402

from src.models.universal_clinical_router_v4 import (  # noqa: E402
    SCRIPTED_COMMON_TASKS,
    cue_aligned_upper_probability,
    evidence_profile,
    linear_head_probability,
    median_low_confidence_gate,
    scripted_multimechanism_probability,
    serialized_head_probability,
)


def test_router_uses_protocol_evidence_not_dataset_identity(c: Check):
    c.eq(
        evidence_profile(
            ("FREE_RECORDING",), has_au=False, has_marlin=False,
            timing_authority="none",
        ),
        "free_asymmetry",
        "free recordings use the frozen 110D head",
    )
    c.eq(
        evidence_profile(
            SCRIPTED_COMMON_TASKS, has_au=True, has_marlin=True,
            timing_authority="recording_task_label",
        ),
        "scripted_multimechanism",
        "scripted evidence enables AU and MARLIN phenotype heads",
    )
    c.eq(
        evidence_profile(
            ("BROW_RAISE", "EYE_GENTLE", "EYE_FORCEFUL"),
            has_au=False, has_marlin=False,
            timing_authority="external_prompt",
        ),
        "cue_aligned_upper",
        "authenticated external cues enable the upper-face sequence head",
    )
    signature = inspect.signature(evidence_profile)
    c.true("source" not in signature.parameters, "dataset/source identity is not an input")


def test_router_fails_closed_on_ambiguous_or_missing_modalities(c: Check):
    c.raises(
        lambda: evidence_profile(
            SCRIPTED_COMMON_TASKS, has_au=True, has_marlin=False,
            timing_authority="recording_task_label",
        ),
        ValueError,
        "partial scripted modality sets cannot silently change the estimator",
    )
    c.raises(
        lambda: evidence_profile(
            ("BROW_RAISE",), has_au=False, has_marlin=False,
            timing_authority="visual_peak",
        ),
        ValueError,
        "visual motion cannot authenticate a prompted action",
    )


def test_median_gate_changes_only_low_confidence_clinical_rows(c: Check):
    clinical = np.asarray([0.10, 0.30, 0.50, 0.90], dtype=np.float64)
    marlin = np.asarray([
        [0.99, 0.20, 0.80, 0.01],
        [0.01, 0.60, 0.70, 0.99],
        [0.50, 0.40, 0.90, 0.50],
    ], dtype=np.float64)
    result = median_low_confidence_gate(clinical, marlin)
    c.true(
        np.allclose(result, np.asarray([0.10, 0.40, 0.80, 0.90])),
        "clinical extremes remain fixed and uncertain rows use MARLIN median",
    )
    clinical[1] = 0.99
    marlin[0, 1] = 0.99
    c.true(
        np.allclose(result, np.asarray([0.10, 0.40, 0.80, 0.90])),
        "fusion output is an immutable snapshot",
    )


def test_linear_head_probability_is_exact_and_fail_closed(c: Check):
    values = np.asarray([[1.0, 4.0], [3.0, 8.0]], dtype=np.float64)
    result = linear_head_probability(
        values,
        mean=np.asarray([1.0, 2.0], dtype=np.float64),
        scale=np.asarray([2.0, 2.0], dtype=np.float64),
        coefficient=np.asarray([1.0, -0.5], dtype=np.float64),
        intercept=0.25,
    )
    expected_logit = np.asarray([-0.25, -0.25], dtype=np.float64)
    c.true(np.allclose(result, 1.0 / (1.0 + np.exp(-expected_logit))),
           "linear head applies train-fold scaler then logistic score")
    c.raises(
        lambda: linear_head_probability(
            values.astype(np.float32),
            mean=np.zeros(2), scale=np.ones(2),
            coefficient=np.ones(2), intercept=0.0,
        ),
        ValueError,
        "public head rejects a different numeric schema",
    )


def _head(coefficient, *, dimension=3, selected=(0, 2)):
    return {
        "input_dimension": dimension,
        "selected_indices": list(selected),
        "scaler_center": [0.0] * len(selected),
        "scaler_scale": [1.0] * len(selected),
        "coefficient": list(coefficient),
        "intercept": 0.0,
    }


def test_serialized_head_binds_selection_and_rejects_open_schema(c: Check):
    values = np.asarray([[2.0, 99.0, -1.0]], dtype=np.float64)
    probability = serialized_head_probability(values, _head((1.0, 2.0)))
    c.true(np.allclose(probability, [0.5]), "selected dimensions are applied exactly")
    malformed = dict(_head((1.0, 2.0)), extra="drift")
    c.raises(
        lambda: serialized_head_probability(values, malformed), ValueError,
        "serialized estimator documents use a closed schema",
    )


def test_scripted_route_combines_mechanisms_then_gates_uncertainty(c: Check):
    zero = _head((0.0, 0.0))
    positive = dict(zero, intercept=2.0)
    negative = dict(zero, intercept=-2.0)
    artifact = {
        "clinical_heads": {
            "post_stroke_asymmetry_mean110": zero,
            "als_oromotor_robust_pool1600": zero,
            "fusion": "maximum_probability",
        },
        "marlin_heads": [
            {"representation": "r", "top_k": index + 1, "c": 1.0,
             "targets": [{"phenotype": "als", "head": positive},
                         {"phenotype": "post_stroke", "head": negative}]}
            for index in range(18)
        ],
        "gate": {
            "clinical_center": 0.5, "radius": 0.3,
            "inside": "median_of_18_marlin_phenotype_max_probabilities",
            "outside": "clinical_probability",
        },
    }
    values = np.zeros((1, 3), dtype=np.float64)
    probability = scripted_multimechanism_probability(
        landmark_original=values, landmark_mirrored=values,
        au_values=values, marlin_representations={"r": values},
        artifact=artifact,
    )
    c.true(probability[0] > 0.8, "MARLIN rescues an uncertain clinical score")


def test_cue_route_averages_mirrors_then_fixed_heads(c: Check):
    head = _head((1.0, 0.0))
    artifact = {
        "heads": [{"name": "a", "actions": ["x"], "family": "summary",
                   "head": head},
                  {"name": "b", "actions": ["x"], "family": "summary",
                   "head": head}],
        "probability_weights": [0.25, 0.75],
        "decision_threshold": 0.5,
    }
    original = {"a": np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64),
                "b": np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64)}
    mirrored = {"a": np.asarray([[-2.0, 0.0, 0.0]], dtype=np.float64),
                "b": np.asarray([[-2.0, 0.0, 0.0]], dtype=np.float64)}
    probability = cue_aligned_upper_probability(original, mirrored, artifact)
    c.true(np.allclose(probability, [0.5]), "each head averages mirror probabilities")


if __name__ == "__main__":
    run_all("test_universal_clinical_router_v4", dict(globals()))
