"""Frozen feature contracts for the 110D-Generalization v1 candidates."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_landmark_classical import ClassicalDataset  # noqa: E402
from scripts.run_mirror_invariant_110d import (  # noqa: E402
    CANDIDATE as MIRROR_CANDIDATE,
    FIXED_C,
    FIXED_MAX_ITER,
    FIXED_RANDOM_STATE,
    FIXED_SOLVER,
    FIXED_THRESHOLD,
    build_development_matrices,
    mirror_dynamic_features,
    run_fixed_inner_oof,
)
from src.preprocessing.clinical_dynamics import (  # noqa: E402
    CLINICAL_DYNAMICS_DIM,
    clinical_dynamics_feature_names,
    clinical_dynamics_feature_vector,
)
from src.preprocessing.generalization_110d import (  # noqa: E402
    ACTION_PHASE_PROXY_204D,
    ACTION_PROXY_168D,
    CANDIDATE_REGISTRY,
    LANDMARK_MI_110D,
    PHASE_PROXY_DIM,
    candidate_feature_names,
    candidate_feature_vector,
    phase_proxy_feature_vector,
)
from src.preprocessing.trajectory_features import (  # noqa: E402
    LANDMARK_BILATERAL_PAIRS,
    LANDMARK_DIM,
    trajectory_feature_names,
    trajectory_feature_set,
)
from _testlib import Check, run_all  # noqa: E402


REGION_PAIRS = (
    ("eye", LANDMARK_BILATERAL_PAIRS[:3]),
    ("brow", LANDMARK_BILATERAL_PAIRS[3:4]),
    ("mouth", LANDMARK_BILATERAL_PAIRS[4:]),
)
PHASE_STAT_NAMES = (
    "mean_bilateral_excursion",
    "absolute_excursion_asymmetry",
    "mean_bilateral_peak_velocity_per_second",
)


def _recording(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4, 32, 95)).astype(np.float32)
    mask = np.ones((4, 32), dtype=bool)
    mask[0, 7] = False
    features[0, 7] = 0.0
    timestamps = np.stack([
        window * 10.0 + np.arange(32, dtype=np.float64) * 0.5
        for window in range(4)
    ])
    source_indices = np.stack([
        window * 100 + np.arange(32, dtype=np.int64)
        for window in range(4)
    ])
    return features, mask, timestamps, source_indices


def _synthetic_dataset() -> ClassicalDataset:
    count = 20
    features = np.zeros((count, 4, 32, 95), dtype=np.float32)
    masks = np.ones((count, 4, 32), dtype=bool)
    timestamps = np.stack([
        np.stack([
            window * 10.0 + np.arange(32, dtype=np.float64) * 0.1
            for window in range(4)
        ])
        for _ in range(count)
    ])
    source_indices = np.stack([
        np.stack([
            window * 100 + np.arange(32, dtype=np.int64)
            for window in range(4)
        ])
        for _ in range(count)
    ])
    labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
    ramp = np.arange(32, dtype=np.float32)
    for index, label in enumerate(labels):
        first = 1.0 if label else 0.1
        second = 0.2 if label else 0.1
        features[index, :, :, 72] = first * ramp
        features[index, :, :, 73] = second * ramp
        features[index, :, :, 74] = np.abs(
            features[index, :, :, 72] - features[index, :, :, 73]
        )
        features[index, :, :, 90] = first * ramp
        features[index, :, :, 91] = second * ramp
    return ClassicalDataset(
        features=features,
        valid_masks=masks,
        timestamps=timestamps,
        source_frame_indices=source_indices,
        nuisance=np.zeros((count, 9), dtype=np.float64),
        labels=labels,
        group_ids=np.asarray([
            f"grp_{index:064x}" for index in range(count)
        ]),
        recording_ids=tuple(
            f"rec_{index:064x}" for index in range(count)
        ),
    )


def _new_110_prepared(dataset: ClassicalDataset):
    reference = build_development_matrices(dataset)
    rows: dict[str, list[np.ndarray]] = {
        "original": [],
        "mirrored": [],
        "remirrored": [],
    }
    for global_index in reference.development_indices.tolist():
        raw = dataset.features[global_index]
        mirrored = mirror_dynamic_features(raw)
        remirrored = mirror_dynamic_features(mirrored)
        temporal = (
            dataset.valid_masks[global_index],
            dataset.timestamps[global_index],
            dataset.source_frame_indices[global_index],
        )
        for key, view in (
            ("original", raw),
            ("mirrored", mirrored),
            ("remirrored", remirrored),
        ):
            rows[key].append(candidate_feature_vector(
                LANDMARK_MI_110D, view, *temporal
            ))
    prepared = replace(
        reference,
        original=np.stack(rows["original"]),
        mirrored=np.stack(rows["mirrored"]),
        remirrored=np.stack(rows["remirrored"]),
    )
    return reference, prepared


def test_registry_and_feature_names_are_exact_and_ordered(c: Check):
    expected_registry = (
        ("landmark_mi_110d", 110),
        ("landmark_mi_110d_action_proxy_168d", 168),
        ("landmark_mi_110d_action_phase_proxy_204d", 204),
    )
    c.eq(tuple(CANDIDATE_REGISTRY.items()), expected_registry,
         "candidate order and dimensions are preregistered")

    landmark_names = trajectory_feature_names("landmark")
    action_names = tuple(
        f"action_proxy__{name}" for name in clinical_dynamics_feature_names()
    )
    phase_names = tuple(
        f"phase_proxy__window_{window}__{region}__{statistic}"
        for window in range(4)
        for region, _ in REGION_PAIRS
        for statistic in PHASE_STAT_NAMES
    )
    c.eq(candidate_feature_names(LANDMARK_MI_110D), landmark_names,
         "110D names are exactly the frozen champion names")
    c.eq(candidate_feature_names(ACTION_PROXY_168D),
         landmark_names + action_names,
         "168D appends only the explicitly named 58D Action proxy")
    c.eq(candidate_feature_names(ACTION_PHASE_PROXY_204D),
         landmark_names + action_names + phase_names,
         "204D appends the fixed window-region-statistic Phase proxy order")
    for candidate, dimension in expected_registry:
        names = candidate_feature_names(candidate)
        c.eq(len(names), dimension, f"{candidate} has one name per feature")
        c.eq(len(set(names)), dimension, f"{candidate} names are unique")


def test_vectors_concatenate_only_the_frozen_blocks_and_are_finite(c: Check):
    arrays = _recording(11)
    frozen_110 = trajectory_feature_set("landmark", *arrays)
    frozen_58 = clinical_dynamics_feature_vector(*arrays)
    vector_110 = candidate_feature_vector(LANDMARK_MI_110D, *arrays)
    vector_168 = candidate_feature_vector(ACTION_PROXY_168D, *arrays)
    vector_204 = candidate_feature_vector(ACTION_PHASE_PROXY_204D, *arrays)
    c.eq(LANDMARK_DIM, 110, "champion dimension remains frozen")
    c.eq(CLINICAL_DYNAMICS_DIM, 58, "Action proxy dimension remains frozen")
    c.eq(PHASE_PROXY_DIM, 36, "Phase proxy is exactly 36D")
    c.true(np.array_equal(vector_110, frozen_110),
           "new 110D path delegates byte-for-byte to the champion extractor")
    c.true(np.array_equal(vector_168[:110], frozen_110),
           "168D starts with the exact champion row")
    c.true(np.array_equal(vector_168[110:], frozen_58),
           "168D appends the exact frozen clinical-dynamics row")
    c.true(np.array_equal(vector_204[:168], vector_168),
           "204D does not alter either simpler block")
    for vector, dimension in ((vector_110, 110), (vector_168, 168), (vector_204, 204)):
        c.eq(vector.shape, (dimension,), "candidate vector dimension is exact")
        c.true(np.isfinite(vector).all(), "candidate vector is finite")


def test_phase_proxy_order_and_window_position_are_preserved(c: Check):
    features, mask, timestamps, source_indices = _recording()
    features.fill(0.0)
    mask.fill(True)
    ramp = np.arange(32, dtype=np.float32)
    expected: list[float] = []
    for window in range(4):
        for region_index, (_, pairs) in enumerate(REGION_PAIRS, start=1):
            slope = float((window + 1) * region_index)
            for _, first, second in pairs:
                features[window, :, first] = slope * ramp
                features[window, :, second] = 0.5 * slope * ramp
            expected.extend((23.25 * slope, 15.5 * slope, 1.5 * slope))
    phase = candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        features,
        mask,
        timestamps,
        source_indices,
    )[168:]
    c.true(np.allclose(phase, np.asarray(expected), atol=1e-12),
           "36D order is window then eye/brow/mouth then three fixed summaries")

    rolled_features = np.roll(features, shift=1, axis=0)
    rolled_phase = candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        rolled_features,
        mask,
        timestamps,
        source_indices,
    )[168:]
    c.true(np.array_equal(
        rolled_phase.reshape(4, 9),
        np.roll(phase.reshape(4, 9), shift=1, axis=0),
    ), "moving a recording window moves its intact nine-feature slot")
    c.true(not np.array_equal(rolled_phase, phase),
           "window position is represented instead of pooled away")


def test_phase_velocity_never_crosses_window_or_detector_gaps(c: Check):
    features, mask, timestamps, source_indices = _recording()
    features.fill(0.0)
    mask.fill(True)
    pair_channels = {
        channel
        for _, first, second in LANDMARK_BILATERAL_PAIRS
        for channel in (first, second)
    }
    for window, level in enumerate((0.0, 100.0, 200.0, 300.0)):
        for channel in pair_channels:
            features[window, :, channel] = level
    features[0, 2:, 72] = 1000.0
    mask[0, 1] = False
    phase = candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        features,
        mask,
        timestamps,
        source_indices,
    )[168:].reshape(4, 3, 3)
    c.true(np.array_equal(phase[..., 2], np.zeros((4, 3))),
           "peak velocity never bridges a window boundary or invalid detector gap")


def test_phase_rejects_integer_timestamps_that_collapse_in_float64(c: Check):
    features, mask, _, source_indices = _recording()
    features.fill(0.0)
    ramp = np.arange(32, dtype=np.float32)
    for _, first, second in LANDMARK_BILATERAL_PAIRS:
        features[..., first] = ramp
        features[..., second] = ramp

    ordinary = np.stack([
        window * 100 + np.arange(32, dtype=np.int64)
        for window in range(4)
    ])
    phase = phase_proxy_feature_vector(
        features, mask, ordinary, source_indices
    ).reshape(4, 3, 3)
    c.true(np.array_equal(phase[..., 2], np.ones((4, 3))),
           "exactly representable integer seconds retain unit peak velocity")

    precision_boundary = np.stack([
        np.int64(2**53 + window * 1000) + np.arange(32, dtype=np.int64)
        for window in range(4)
    ])
    c.raises(lambda: phase_proxy_feature_vector(
        features, mask, precision_boundary, source_indices
    ), ValueError,
        "timestamps must remain strictly increasing after float64 normalization")


def test_added_blocks_are_exactly_capture_side_swap_invariant(c: Check):
    features, mask, timestamps, source_indices = _recording(17)
    swapped = features.copy()
    for _, first, second in LANDMARK_BILATERAL_PAIRS:
        swapped[..., first], swapped[..., second] = (
            features[..., second], features[..., first]
        )
    original = candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        features,
        mask,
        timestamps,
        source_indices,
    )
    capture_swapped = candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        swapped,
        mask,
        timestamps,
        source_indices,
    )
    c.true(np.array_equal(original[110:168], capture_swapped[110:168]),
           "the reused 58D suffix is exactly direction-free")
    c.true(np.array_equal(original[168:], capture_swapped[168:]),
           "the new 36D suffix is exactly direction-free")


def test_extractor_fails_closed_on_candidate_and_schema_drift(c: Check):
    features, mask, timestamps, source_indices = _recording()
    c.raises(lambda: candidate_feature_names("unknown"), ValueError,
             "only the three frozen candidate names are accepted")
    c.raises(lambda: candidate_feature_vector(
        "unknown", features, mask, timestamps, source_indices
    ), ValueError, "unknown candidates fail before extraction")
    c.raises(lambda: candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        features[..., :-1],
        mask,
        timestamps,
        source_indices,
    ), ValueError, "the 95-column input schema is mandatory")
    c.raises(lambda: candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        features,
        mask.astype(np.uint8),
        timestamps,
        source_indices,
    ), ValueError, "mask provenance is mandatory")
    nonfinite = features.copy()
    nonfinite[0, 0, 72] = np.nan
    c.raises(lambda: candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        nonfinite,
        mask,
        timestamps,
        source_indices,
    ), ValueError, "nonfinite valid geometry fails closed")
    missing_window = mask.copy()
    missing_window[2] = False
    c.raises(lambda: candidate_feature_vector(
        ACTION_PHASE_PROXY_204D,
        features,
        missing_window,
        timestamps,
        source_indices,
    ), ValueError, "every one of the four Phase-proxy slots requires evidence")


def test_new_110_path_is_bit_identical_through_mirror_oof(c: Check):
    dataset = _synthetic_dataset()
    reference, prepared = _new_110_prepared(dataset)
    c.eq(
        (FIXED_C, FIXED_SOLVER, FIXED_MAX_ITER, FIXED_RANDOM_STATE, FIXED_THRESHOLD),
        (0.01, "liblinear", 2000, 0, 0.5),
        "the shared frozen model settings are unchanged",
    )
    for field in ("original", "mirrored", "remirrored"):
        c.true(np.array_equal(getattr(prepared, field), getattr(reference, field)),
               f"new 110D {field} rows are bit-identical to the reference path")
    c.true(np.array_equal(prepared.remirrored, prepared.original),
           "new 110D mirror rows retain exact one-to-one involutive pairing")
    protected = set(reference.protected_indices.tolist())
    c.true(set(reference.extraction_indices).isdisjoint(protected),
           "the synthetic equivalence fixture extracts development rows only")

    reference_oof = run_fixed_inner_oof(dataset, reference, MIRROR_CANDIDATE)
    new_path_oof = run_fixed_inner_oof(dataset, prepared, MIRROR_CANDIDATE)
    c.true(np.array_equal(new_path_oof.probabilities, reference_oof.probabilities),
           "new 110D OOF probabilities are bit-identical to the frozen runner")
    c.eq(new_path_oof.audit_events, reference_oof.audit_events,
         "fit and prediction index audits are unchanged")
    c.eq(new_path_oof.max_mirror_probability_error,
         reference_oof.max_mirror_probability_error,
         "symmetric inference has the identical mirror error")
    for event in new_path_oof.audit_events:
        c.true(set(event.indices).isdisjoint(protected),
               "equivalence fitting and prediction never touch protected rows")


if __name__ == "__main__":
    run_all("test_110d_generalization_features", dict(globals()))
