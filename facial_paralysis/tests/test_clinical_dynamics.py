"""Frozen, direction-free clinical landmark dynamics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.clinical_dynamics import (  # noqa: E402
    CLINICAL_DYNAMICS_DIM,
    CLINICAL_DYNAMICS_GLOBAL_CHANNELS,
    clinical_dynamics_feature_names,
    clinical_dynamics_feature_vector,
)
from src.preprocessing.trajectory_features import LANDMARK_BILATERAL_PAIRS  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((4, 32, 95), dtype=np.float32)
    mask = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([
        window * 10.0 + np.arange(32, dtype=np.float64) * 0.1
        for window in range(4)
    ])
    source_indices = np.stack([
        window * 100 + np.arange(32, dtype=np.int64)
        for window in range(4)
    ])
    return features, mask, timestamps, source_indices


def _extract(arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return clinical_dynamics_feature_vector(*arrays)


def test_contract_has_exact_58_features_in_clinical_order(c: Check):
    names = clinical_dynamics_feature_names()
    c.eq(CLINICAL_DYNAMICS_DIM, 58, "frozen candidate stays compact")
    c.eq(len(names), CLINICAL_DYNAMICS_DIM, "every output has one ordered name")
    c.eq(len(set(names)), len(names), "feature names are unique")
    c.eq(names[0], "fissure_h_mesh33_vs_mesh263__correlation",
         "eye pair dynamics come first")
    c.eq(names[-1], "mouth_open__max_abs_velocity_per_second",
         "global mouth dynamics close the contract")
    c.eq(tuple(index for _, index in CLINICAL_DYNAMICS_GLOBAL_CHANNELS),
         (74, 78, 81, 84, 88, 92, 93, 94),
         "only explicit asymmetry and global clinical channels are added")


def test_direction_free_vector_is_exactly_invariant_to_capture_side_swap(c: Check):
    arrays = list(_recording())
    rng = np.random.default_rng(7)
    arrays[0][..., 72:] = rng.normal(size=(4, 32, 23))
    original = _extract(tuple(arrays))
    mirrored = arrays[0].copy()
    for _, first, second in LANDMARK_BILATERAL_PAIRS:
        mirrored[..., first], mirrored[..., second] = (
            arrays[0][..., second], arrays[0][..., first]
        )
    swapped = _extract((mirrored, arrays[1], arrays[2], arrays[3]))
    c.true(np.allclose(original, swapped, atol=1e-12),
           "unknown capture mirroring cannot change this candidate")


def test_pair_features_encode_correlation_ratio_absolute_lag_and_excursion(c: Check):
    features, mask, timestamps, source_indices = _recording()
    ramp = np.arange(32, dtype=np.float32)
    features[..., 72] = ramp
    features[..., 73] = 0.5 * ramp
    vector = _extract((features, mask, timestamps, source_indices))
    c.true(np.allclose(vector[:3], (1.0, 0.5, 0.0), atol=1e-12),
           "first pair reports correlation, invariant amplitude ratio, and absolute lag")
    c.true(np.allclose(vector[3:7], (23.25, 15.5, 7.5, 5.0), atol=1e-12),
           "first pair reports symmetric excursion and velocity summaries")


def test_absolute_lag_is_nonzero_bounded_and_freezes_endpoint_valid_semantics(c: Check):
    features, mask, timestamps, source_indices = _recording()
    rng = np.random.default_rng(19)
    signal = rng.normal(size=32).astype(np.float32)
    delayed = np.zeros(32, dtype=np.float32)
    delayed[3:] = signal[:-3]
    features[..., 72] = signal
    features[..., 73] = delayed
    vector = _extract((features, mask, timestamps, source_indices))
    c.true(np.isclose(vector[2], 0.3, atol=1e-12),
           "the first pair reports a known three-frame lag in seconds")
    c.true(0.0 <= vector[2] <= 0.5 + 1e-12,
           "the frozen lag search never exceeds five frames")

    delayed_six = np.zeros(32, dtype=np.float32)
    delayed_six[6:] = signal[:-6]
    features[..., 73] = delayed_six
    six_frame_probe = _extract((features, mask, timestamps, source_indices))
    c.true(six_frame_probe[2] <= 0.5 + 1e-12,
           "a six-frame signal cannot expand the frozen five-frame search")
    c.true(not np.isclose(six_frame_probe[2], 0.6, atol=1e-12),
           "the unavailable six-frame lag is never reported")

    # Freeze the documented v1 limitation: lagged endpoints may be valid even
    # when an intermediate detector frame is invalid.
    features.fill(0.0)
    mask.fill(False)
    for window in range(4):
        mask[window, (0, 2, 4, 6)] = True
        features[window, (0, 2, 4), 72] = (1.0, 2.0, 4.0)
        features[window, (2, 4, 6), 73] = (1.0, 2.0, 4.0)
    endpoint_valid = _extract((features, mask, timestamps, source_indices))
    c.true(np.isclose(endpoint_valid[2], 0.2, atol=1e-12),
           "v1 lag uses valid endpoints without intermediate continuity")


def test_velocity_never_bridges_windows_or_detector_gaps(c: Check):
    features, mask, timestamps, source_indices = _recording()
    # mouth_open is constant inside every window but jumps between windows.
    for window, level in enumerate((0.0, 10.0, 20.0, 30.0)):
        features[window, :, 94] = level
    # This large jump straddles an invalid detector frame and must be ignored.
    features[0, :, 93] = 0.0
    features[0, 2:, 93] = 100.0
    mask[0, 1] = False
    vector = _extract((features, mask, timestamps, source_indices))
    names = clinical_dynamics_feature_names()
    c.eq(float(vector[names.index("mouth_open__max_abs_velocity_per_second")]), 0.0,
         "window boundaries do not create velocity")
    c.eq(float(vector[names.index("mouth_width__max_abs_velocity_per_second")]), 0.0,
         "detector gaps do not create velocity")


def test_extractor_fails_closed_on_schema_drift(c: Check):
    features, mask, timestamps, source_indices = _recording()
    c.raises(lambda: clinical_dynamics_feature_vector(
        features[..., :-1], mask, timestamps, source_indices
    ), ValueError, "the frozen 95-column input is mandatory")
    c.raises(lambda: clinical_dynamics_feature_vector(
        features, mask.astype(np.uint8), timestamps, source_indices
    ), ValueError, "mask provenance is mandatory")


if __name__ == "__main__":
    run_all("test_clinical_dynamics", dict(globals()))
