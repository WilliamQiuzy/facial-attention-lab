"""Pure contracts for canonical 30-Hz dynamic-landmark SSL packets."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)
from src.preprocessing.semantic_landmarks import (  # noqa: E402
    SEMANTIC23_FEATURE_NAMES,
    SEMANTIC23_SCHEMA,
)
from src.pretraining import dynamic_landmark_ssl_bridge as bridge_core  # noqa: E402
from src.pretraining.dynamic_landmark_ssl_bridge import (  # noqa: E402
    BridgePolicy,
    packetize_mayo_trajectory,
    packetize_ravdess_trajectory,
    uniform_floor_starts,
)


# Exact audited row-count distribution for all 2,452 RAVDESS tracking trials.
# It is compact test evidence for the frozen 88..191-frame inventory and avoids
# coupling this pure suite to the local external-data directory.
_RAVDESS_LENGTH_COUNTS = {
    88: 1, 89: 1, 90: 1, 91: 4, 92: 8, 93: 10, 94: 20, 95: 14,
    96: 25, 97: 35, 98: 27, 99: 34, 100: 33, 101: 32, 102: 40,
    103: 55, 104: 60, 105: 67, 106: 71, 107: 63, 108: 67,
    109: 64, 110: 63, 111: 67, 112: 63, 113: 50, 114: 48,
    115: 53, 116: 46, 117: 34, 118: 35, 119: 30, 120: 41,
    121: 22, 122: 39, 123: 46, 124: 51, 125: 36, 126: 25,
    127: 35, 128: 48, 129: 38, 130: 40, 131: 40, 132: 44,
    133: 37, 134: 43, 135: 32, 136: 35, 137: 49, 138: 41,
    139: 45, 140: 41, 141: 27, 142: 32, 143: 30, 144: 21,
    145: 24, 146: 23, 147: 19, 148: 17, 149: 22, 150: 16,
    151: 12, 152: 9, 153: 16, 154: 13, 155: 11, 156: 11,
    157: 13, 158: 8, 159: 7, 160: 8, 161: 17, 162: 7, 163: 2,
    165: 4, 166: 6, 167: 1, 168: 3, 169: 4, 171: 2, 173: 1,
    174: 2, 175: 3, 177: 1, 178: 1, 180: 4, 182: 1, 183: 2,
    185: 1, 189: 1, 191: 1,
}


def _trajectory(length: int, width: int):
    values = np.arange(length * width, dtype=np.float32).reshape(length, width)
    features = np.remainder(values, np.float32(101.0)) / np.float32(100.0)
    valid_mask = np.ones(length, dtype=np.bool_)
    canonical_indices = np.arange(length, dtype=np.int64)
    original_source_indices = np.arange(length, dtype=np.int64) * 2 + 1
    original_timestamps = np.arange(length, dtype=np.float64) / 29.97
    return (
        features,
        valid_mask,
        canonical_indices,
        original_source_indices,
        original_timestamps,
    )


def _ravdess(length: int, **changes):
    fields = dict(zip(
        ("features", "valid_mask", "canonical_frame_indices",
         "original_source_frame_indices", "original_timestamps"),
        _trajectory(length, 23),
    ))
    fields.update({
        "feature_schema": SEMANTIC23_SCHEMA,
        "feature_names": SEMANTIC23_FEATURE_NAMES,
    })
    fields.update(changes)
    return packetize_ravdess_trajectory(**fields)


def _mayo(length: int, **changes):
    fields = dict(zip(
        ("features", "valid_mask", "canonical_frame_indices",
         "original_source_frame_indices", "original_timestamps"),
        _trajectory(length, 95),
    ))
    fields.update({
        "feature_schema": DYNAMIC_FEATURE_SCHEMA,
        "feature_names": DYNAMIC_FEATURE_NAMES,
    })
    fields.update(changes)
    return packetize_mayo_trajectory(**fields)


def test_policy_and_all_frozen_ravdess_lengths_packetize(c: Check):
    policy = BridgePolicy()
    c.eq(policy.sample_rate_hz, 30.0)
    c.eq(policy.window_length, 32)
    c.eq(policy.ravdess_packets_per_trial, 1)
    c.eq(policy.mayo_packets_per_recording, 16)
    c.eq(policy.selection, "uniform_floor_v1")

    starts = uniform_floor_starts(88, count=4, window=32)
    c.true(bool(np.array_equal(starts, np.asarray([0, 18, 37, 56], np.int64))))
    lengths = [
        length
        for length, count in _RAVDESS_LENGTH_COUNTS.items()
        for _ in range(count)
    ]
    c.eq(len(lengths), 2_452)
    c.eq(sum(lengths), 299_854)
    expected_t = np.arange(32, dtype=np.float32) / np.float32(30.0)
    expected_i = np.arange(32, dtype=np.int64)
    for length in lengths:
        bundle, mapping = _ravdess(length)
        c.eq(bundle.features.shape, (1, 4, 32, 23))
        c.eq(bundle.features.dtype, np.dtype(np.float32))
        c.eq(bundle.valid_mask.dtype, np.dtype(np.bool_))
        c.true(bool(np.array_equal(
            bundle.timestamps,
            np.broadcast_to(expected_t, bundle.timestamps.shape),
        )), "every RAVDESS window uses the exact local float32 timeline")
        c.true(bool(np.array_equal(
            bundle.source_frame_indices,
            np.broadcast_to(expected_i, bundle.source_frame_indices.shape),
        )), "RAVDESS bundle indices are local canonical positions")
        c.eq(mapping.window_starts.shape, (1, 4))
        c.eq(int(mapping.window_starts[0, 0]), 0)
        c.eq(int(mapping.window_starts[0, -1]), length - 32)


def test_mayo_packets_use_exact_local_axes_and_quartile_layout(c: Check):
    length = 257
    features, valid, canonical, source, source_times = _trajectory(length, 95)
    starts = uniform_floor_starts(length, count=64, window=32)
    gap = int(starts[20] + 7)
    valid[gap] = False
    features[gap] = np.float32(7.0)

    calls: list[tuple[int, ...]] = []
    original_adapter = bridge_core.clinical23_v2_to_semantic23

    def checked_adapter(values: np.ndarray) -> np.ndarray:
        calls.append(values.shape)
        return original_adapter(values)

    bridge_core.clinical23_v2_to_semantic23 = checked_adapter
    try:
        bundle, mapping = _mayo(
            length,
            features=features,
            valid_mask=valid,
            canonical_frame_indices=canonical,
            original_source_frame_indices=source,
            original_timestamps=source_times,
        )
    finally:
        bridge_core.clinical23_v2_to_semantic23 = original_adapter

    c.eq(calls, [(length, 23)],
         "Mayo final23 must pass through the explicit clinical23 adapter")
    c.eq(bundle.features.shape, (16, 4, 32, 95))
    c.eq(bundle.valid_mask.shape, (16, 4, 32))
    expected_packet_starts = np.stack(
        tuple(starts[offset:offset + 16] for offset in (0, 16, 32, 48)),
        axis=1,
    )
    c.true(bool(np.array_equal(mapping.window_starts, expected_packet_starts)))
    expected_t = np.arange(32, dtype=np.float32) / np.float32(30.0)
    expected_i = np.arange(32, dtype=np.int64)
    c.true(bool(np.array_equal(
        bundle.timestamps,
        np.broadcast_to(expected_t, bundle.timestamps.shape),
    )), "every long-recording window has the exact local float32 timeline")
    c.true(bool(np.array_equal(
        bundle.source_frame_indices,
        np.broadcast_to(expected_i, bundle.source_frame_indices.shape),
    )), "every window has exact local canonical indices, never source offsets")
    gap_slots = mapping.original_canonical_frame_indices == gap
    c.true(bool(gap_slots.any()), "the selected fixed grid includes the synthetic gap")
    c.true(bool((~bundle.valid_mask[gap_slots]).all()))
    c.true(bool((bundle.features[gap_slots] == 0.0).all()),
           "a missing canonical row is zero/False and is never compressed")
    next_slots = mapping.original_canonical_frame_indices == gap + 1
    c.true(bool(next_slots.any()))
    c.true(bool(bundle.valid_mask[next_slots].all()),
           "the observed row after a gap retains its exact grid position")
    c.true(bool(np.array_equal(bundle.features, features[
        mapping.original_canonical_frame_indices
    ] * bundle.valid_mask[..., None])),
        "the full 95d Mayo signal is retained rather than replaced by semantic23")


def test_selection_depends_only_on_length(c: Check):
    length = 257
    first_features, first_mask, *_ = _trajectory(length, 95)
    second_features = np.flip(first_features, axis=0).copy()
    second_mask = np.ones(length, dtype=np.bool_)
    second_mask[::11] = False
    first_bundle, first_mapping = _mayo(
        length, features=first_features, valid_mask=first_mask,
    )
    second_bundle, second_mapping = _mayo(
        length, features=second_features, valid_mask=second_mask,
    )
    c.true(bool(np.array_equal(
        first_mapping.window_starts, second_mapping.window_starts,
    )), "features and masks cannot affect window starts")
    metadata_variants = (
        {"label": 0, "movement": np.zeros(length)},
        {"label": 6, "movement": np.linspace(100.0, -100.0, length)},
    )
    metadata_starts = [
        uniform_floor_starts(length, count=64, window=32)
        for _metadata in metadata_variants
    ]
    c.true(bool(np.array_equal(metadata_starts[0], metadata_starts[1])),
           "labels and movement are outside the length-only selection API")
    parameters = inspect.signature(packetize_mayo_trajectory).parameters
    c.true("label" not in parameters and "movement" not in parameters,
           "content metadata cannot enter the pure packetizer")
    c.eq(first_bundle.features.shape, second_bundle.features.shape)


def test_malformed_trajectories_fail_closed(c: Check):
    c.raises(lambda: _ravdess(31), ValueError, "T<32 must fail closed")
    c.raises(lambda: _ravdess(88, feature_schema="clinical23_v2"), ValueError,
             "wrong RAVDESS schema")
    c.raises(lambda: _ravdess(
        88, feature_names=tuple(reversed(SEMANTIC23_FEATURE_NAMES)),
    ), ValueError, "wrong semantic23 order")
    c.raises(lambda: _ravdess(
        88, features=np.zeros((88, 22), dtype=np.float32),
    ), ValueError, "wrong RAVDESS width")
    c.raises(lambda: _ravdess(
        88, features=_trajectory(88, 23)[0].astype(np.float64),
    ), ValueError, "wrong feature dtype")
    c.raises(lambda: _ravdess(
        88, valid_mask=np.ones(88, dtype=np.uint8),
    ), ValueError, "wrong mask dtype")
    c.raises(lambda: _ravdess(
        88, canonical_frame_indices=np.arange(88, dtype=np.int32),
    ), ValueError, "wrong canonical-index dtype")
    c.raises(lambda: _ravdess(
        88, original_source_frame_indices=np.arange(88, dtype=np.int32),
    ), ValueError, "wrong source-index dtype")
    c.raises(lambda: _ravdess(
        88, original_timestamps=np.arange(88, dtype=np.float32) / 30,
    ), ValueError, "wrong timestamp dtype")
    nonfinite, *_rest = _trajectory(88, 23)
    nonfinite[0, 0] = np.nan
    c.raises(lambda: _ravdess(88, features=nonfinite), ValueError,
             "nonfinite trajectories")
    c.raises(lambda: _mayo(
        128,
        feature_schema="arkit_blendshapes_52_v1",
        feature_names=tuple(f"arkit_{index}" for index in range(52)),
        features=np.zeros((128, 52), dtype=np.float32),
    ), ValueError, "ARKit 52d is auxiliary-only")


def test_forged_bridge_policies_fail_closed(c: Check):
    class BadPolicy(BridgePolicy):
        def __post_init__(self) -> None:
            pass

    subclass_policy = BadPolicy(
        sample_rate_hz=31.0,
        ravdess_packets_per_trial=99,
        selection="anything",
    )
    c.raises(lambda: _ravdess(
        88, policy=subclass_policy,
    ), ValueError, "a subclass cannot override frozen policy validation")

    mutated_policy = BridgePolicy()
    object.__setattr__(mutated_policy, "sample_rate_hz", 31.0)
    object.__setattr__(mutated_policy, "selection", "anything")
    c.raises(lambda: _ravdess(
        88, policy=mutated_policy,
    ), ValueError, "an exact-type policy is revalidated after construction")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_ssl_bridge", dict(globals()))
