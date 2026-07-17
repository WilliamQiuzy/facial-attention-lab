"""Pure contracts for canonical 30-Hz dynamic-landmark SSL packets."""
from __future__ import annotations

import base64
import contextlib
import errno
import hashlib
import hmac
import importlib.util
import inspect
import io
import json
import logging
import os
import stat
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


CLI_SCRIPT = ROOT / "scripts" / "prepare_dynamic_landmark_ssl_inputs.py"


def _load_cli():
    if not CLI_SCRIPT.is_file():
        raise AssertionError("Task 2 bridge CLI is missing")
    spec = importlib.util.spec_from_file_location(
        "prepare_dynamic_landmark_ssl_inputs_test", CLI_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Task 2 bridge CLI cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_private_bundle_publication_apis_are_explicit(c: Check):
    for name in (
        "build_bridge_bundles",
        "verify_bridge_generation",
    ):
        c.true(hasattr(bridge_core, name), f"missing Task 2 API: {name}")


_PRODUCTION_BRIDGE_CONTRACT = {
    "_FROZEN_RAVDESS_TRIAL_COUNT": 2_452,
    "_FROZEN_RAVDESS_ACTOR_COUNT": 24,
    "_FROZEN_RAVDESS_SOURCE_FRAMES": 299_854,
    "_FROZEN_RAVDESS_SAMPLE_COUNT": 2_452,
    "_FROZEN_MAYO_MEDIAPIPE_COUNT": 48,
    "_FROZEN_MAYO_ARKIT_COUNT": 8,
    "_FROZEN_MAYO_CACHE_COUNT": 56,
    "_FROZEN_MAYO_SAMPLE_COUNT": 768,
}


def _set_bridge_contract(values: dict[str, int]) -> None:
    for name, value in values.items():
        setattr(bridge_core, name, value)


def _synthetic_authorizations():
    _set_bridge_contract({
        "_FROZEN_RAVDESS_TRIAL_COUNT": 2,
        "_FROZEN_RAVDESS_ACTOR_COUNT": 2,
        "_FROZEN_RAVDESS_SOURCE_FRAMES": 185,
        "_FROZEN_RAVDESS_SAMPLE_COUNT": 2,
        "_FROZEN_MAYO_MEDIAPIPE_COUNT": 2,
        "_FROZEN_MAYO_ARKIT_COUNT": 1,
        "_FROZEN_MAYO_CACHE_COUNT": 3,
        "_FROZEN_MAYO_SAMPLE_COUNT": 32,
    })
    ravdess_trials = []
    for index, length in enumerate((88, 97)):
        features, valid, _canonical, source, timestamps = _trajectory(length, 23)
        suffix = "a" * 15 + chr(ord("a") + index)
        ravdess_trials.append(SimpleNamespace(
            trial_id=f"trial_{suffix}",
            actor_id=f"actor_{suffix}",
            cache_integrity_id=f"cache_{suffix}",
            cache_sha256=f"{index + 1:x}" * 64,
            cache_size_bytes=100 + index,
            features=features,
            valid_mask=valid,
            timestamps=timestamps,
            frame_indices=source,
            detector_confidence=np.ones(length, dtype=np.float32),
        ))
    ravdess = SimpleNamespace(
        schema=SEMANTIC23_SCHEMA,
        manifest_sha256="a" * 64,
        generation_closure_hmac="b" * 64,
        trial_count=2,
        actor_count=2,
        source_frames=sum(len(item.features) for item in ravdess_trials),
        valid_frames=sum(len(item.features) for item in ravdess_trials),
        expected_trial_count=2,
        expected_actor_count=2,
        trials=tuple(ravdess_trials),
        private_key=b"r" * 32,
        key_file_identity_sha256="1" * 64,
    )

    mayo_recordings = []
    for index, length in enumerate((128, 143)):
        features, valid, canonical, source, timestamps = _trajectory(length, 95)
        mayo_recordings.append(SimpleNamespace(
            recording_id=f"rec_{index + 1:064x}",
            group_id=f"grp_{index + 3:064x}",
            cache_integrity_id=f"cache_{index + 5:064x}",
            cache_sha256=f"{index + 3:x}" * 64,
            cache_size_bytes=200 + index,
            features_30hz=features,
            valid_mask_30hz=valid,
            timestamps_30hz=timestamps,
            source_frame_indices_30hz=source,
            target_frame_indices_30hz=canonical,
        ))
    commitment = {
        "schema": "mayo_cache_generation_commitment_v3",
        "collection_manifest_sha256": "c" * 64,
        "exposure_manifest_sha256": "d" * 64,
        "mediapipe_file_count": 2,
        "arkit_file_count": 1,
        "cache_file_count": 3,
        "cache_tree_aggregate_sha256": "6" * 64,
        "generation_aggregate_sha256": "7" * 64,
        "inventory_counts_sha256": "8" * 64,
        "collection_classification_integrity_id": "agg_" + "9" * 64,
        "exposure_classification_integrity_id": "agg_" + "a" * 64,
    }
    mayo = SimpleNamespace(
        schema="mayo_mediapipe_clinical23_ssl_v2",
        collection_manifest_sha256="c" * 64,
        exposure_manifest_sha256="d" * 64,
        generation_closure_hmac="e" * 64,
        recording_count=2,
        arkit_count=1,
        expected_recording_count=2,
        commitment=commitment,
        recordings=tuple(mayo_recordings),
        private_key=b"m" * 32,
        key_file_identity_sha256="2" * 64,
    )
    return ravdess, mayo


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def test_mode_neutral_bundle_transaction_is_exact_private_and_keyed(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        output = parent / "bridge"
        result = bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        c.eq(result["ravdess"]["sample_count"], 2)
        c.eq(result["mayo"]["sample_count"], 32)
        c.eq(result["ravdess"]["unique_group_count"], 2)
        c.eq(result["mayo"]["source_unit_count"], 2)
        c.eq(result["mayo"]["upstream_cache_count"], 2)
        bundle_paths = (
            output / "bundles" / "ravdess_bundle.npz",
            output / "bundles" / "mayo_bundle.npz",
        )
        for path in (*bundle_paths, output / "bundle_generation.json"):
            c.eq(stat.S_IMODE(path.stat().st_mode), 0o600,
                 "every private generation file is exact mode 0600")
        expected_fields = {
            "features", "valid_mask", "timestamps",
            "source_frame_indices", "group_ids",
        }
        with np.load(bundle_paths[0], allow_pickle=False) as cached:
            c.eq(set(cached.files), expected_fields)
            c.eq(cached["features"].shape, (2, 4, 32, 23))
            c.eq(cached["valid_mask"].dtype, np.dtype(np.bool_))
            c.eq(cached["timestamps"].dtype, np.dtype(np.float32))
            c.eq(cached["source_frame_indices"].dtype, np.dtype(np.int64))
            c.eq(len(np.unique(cached["group_ids"])), 2)
        with np.load(bundle_paths[1], allow_pickle=False) as cached:
            c.eq(set(cached.files), expected_fields)
            c.eq(cached["features"].shape, (32, 4, 32, 95))
            c.eq(len(np.unique(cached["group_ids"])), 2)
            counts = np.unique(cached["group_ids"], return_counts=True)[1]
            c.true(bool(np.array_equal(counts, np.asarray([16, 16]))))
        generation_text = (output / "bundle_generation.json").read_text("utf-8")
        generation = json.loads(generation_text)
        c.eq(set(generation), {
            "schema", "producer_sha256", "stages",
            "dual_stage_closure_sha256", "dual_stage_closure_hmac",
        })
        c.eq(set(generation["dual_stage_closure_hmac"]), {"ravdess", "mayo"})
        unsigned_generation = dict(generation)
        dual_hmac = unsigned_generation.pop("dual_stage_closure_hmac")
        dual_material = (
            b"dynamic-landmark-bridge-dual-stage-keyed-v1\0"
            + json.dumps(
                unsigned_generation, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")
        )
        c.eq(dual_hmac["ravdess"], hmac.new(
            ravdess.private_key, dual_material, hashlib.sha256,
        ).hexdigest(), "RAVDESS key binds the complete dual-stage generation")
        c.eq(dual_hmac["mayo"], hmac.new(
            mayo.private_key, dual_material, hashlib.sha256,
        ).hexdigest(), "Mayo key binds the complete dual-stage generation")
        c.true(generation["stages"]["ravdess"]["closure_hmac"]
               != generation["stages"]["mayo"]["closure_hmac"],
               "each source key attests its own stage closure")
        forbidden = (
            str(parent), "source_sha256", "private_key", "session",
            "patient", "run_mode", "config", "split", "scaler", "receipt",
        )
        c.true(all(value not in generation_text for value in forbidden),
               "mode-neutral closure contains no private locations or run artifacts")

        before = _tree_bytes(output)
        c.raises(lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), FileExistsError, "an existing committed bridge is never overwritten")
        c.eq(_tree_bytes(output), before,
             "failed replacement preserves every committed byte")

        verification = bridge_core.verify_bridge_generation(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        c.true(verification["deterministic"])
        c.eq(verification["bundle_count"], 2)
        c.true(verification["modes_ok"] and verification["privacy_ok"])
        c.true(not any("verify" in path.name or "staging" in path.name
                       for path in parent.iterdir()),
               "determinism is read-only and creates no filesystem sibling")


def test_bundle_transaction_reauthorizes_and_retains_failed_private_staging(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        output = parent / "bridge"
        calls = 0
        original_close = bridge_core.os.close
        close_injected = False

        def changed_ravdess():
            nonlocal calls
            calls += 1
            if calls == 1:
                return ravdess
            changed = dict(vars(ravdess))
            changed["generation_closure_hmac"] = "0" * 64
            return SimpleNamespace(**changed)

        def close_after_primary(descriptor):
            nonlocal close_injected
            original_close(descriptor)
            if calls >= 2 and not close_injected:
                close_injected = True
                raise OSError("synthetic retained staging close failure")

        bridge_core.os.close = close_after_primary
        try:
            observed = _caught(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=changed_ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256="f" * 64,
            ))
        finally:
            bridge_core.os.close = original_close
        c.true(isinstance(observed, RuntimeError))
        c.true("retained" in str(observed) and "indeterminate" in str(observed),
               "failed owned staging is retained for audit instead of deleted")
        c.true(close_injected)
        c.true(
            _linear_exception_chain_contains(
                observed, ValueError, "upstream authorization changed",
            )
            and _linear_exception_chain_contains(
                observed, OSError, "synthetic retained staging close failure",
            ),
            "retained-state error chains primary and staging close failures",
        )
        c.true(not output.exists(), "failed transaction publishes no generation")
        residue = [path for path in parent.iterdir() if ".bridge.staging-" in path.name]
        c.eq(len(residue), 1)
        c.eq(stat.S_IMODE(residue[0].stat().st_mode), 0o700)
        authorizations = 0

        def must_not_reauthorize():
            nonlocal authorizations
            authorizations += 1
            return ravdess

        _assert_failed(c, lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=must_not_reauthorize,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        ), "retained residue blocks retry")
        c.eq(authorizations, 0, "residue blocks retry before live authorization")

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "bridge"
        bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        )
        changed_key = dict(vars(mayo))
        changed_key["private_key"] = b"z" * 32
        changed_key["generation_closure_hmac"] = "9" * 64
        c.raises(lambda: bridge_core.verify_bridge_generation(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: SimpleNamespace(**changed_key),
            producer_sha256="f" * 64,
        ), ValueError, "a changed canonical source key cannot verify old bundles")


def test_bridge_rejects_packets_without_two_valid_mask_spans_per_window(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    broken_trials = list(ravdess.trials)
    first = dict(vars(broken_trials[0]))
    sparse = np.zeros_like(first["valid_mask"])
    sparse[:4] = True
    first["valid_mask"] = sparse
    broken_trials[0] = SimpleNamespace(**first)
    changed = dict(vars(ravdess))
    changed["trials"] = tuple(broken_trials)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "bridge"
        c.raises(lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: SimpleNamespace(**changed),
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        ), ValueError,
        "every packet window needs two non-overlapping contiguous valid spans of four")
        c.true(not output.exists(), "mask-span failure publishes no generation")


def test_total_bundle_size_is_gated_before_publication(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    prepared = bridge_core._prepare_bridge_generation(
        ravdess, mayo, producer_sha256="f" * 64,
    )
    individual = (
        len(prepared.ravdess.bundle_bytes), len(prepared.mayo.bundle_bytes),
    )
    limit = max(individual) + 1
    c.true(sum(individual) > limit,
           "synthetic fixture separates per-file and aggregate bounds")
    original_limit = bridge_core._MAX_BUNDLE_BYTES
    bridge_core._MAX_BUNDLE_BYTES = limit
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bridge"
            c.raises(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256="f" * 64,
            ), ValueError, "aggregate size is rejected before atomic publication")
            c.true(not output.exists())
    finally:
        bridge_core._MAX_BUNDLE_BYTES = original_limit


def test_post_publish_fault_is_indeterminate_and_never_cleaned_or_retried(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    original_validate = bridge_core._validate_generation_fd
    validations = 0

    def fail_after_publish(root_fd, expected):
        nonlocal validations
        validations += 1
        result = original_validate(root_fd, expected)
        if validations == 3:
            raise OSError("fault after no-replace publication")
        return result

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "bridge"
        bridge_core._validate_generation_fd = fail_after_publish
        try:
            c.raises(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256="f" * 64,
            ), OSError, "post-publication fault is surfaced as indeterminate")
        finally:
            bridge_core._validate_generation_fd = original_validate
        c.true(output.is_dir(),
               "a post-publication fault never deletes the possibly committed generation")
        c.raises(lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        ), FileExistsError, "indeterminate publication cannot be retried in place")


def test_verify_determinism_is_two_pass_in_memory_and_zero_write(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "bridge"
        bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        original_build = bridge_core._build_bridge_bundles_at
        original_write = bridge_core.os.write
        calls = {"ravdess": 0, "mayo": 0, "writes": 0}

        def forbidden_build(*_args, **_kwargs):
            raise AssertionError("read-only verification must not build a sibling")

        def forbidden_write(*_args, **_kwargs):
            calls["writes"] += 1
            raise AssertionError("read-only verification must not write")

        def authorize_ravdess():
            calls["ravdess"] += 1
            return ravdess

        def authorize_mayo():
            calls["mayo"] += 1
            return mayo

        bridge_core._build_bridge_bundles_at = forbidden_build
        bridge_core.os.write = forbidden_write
        try:
            result = bridge_core.verify_bridge_generation(
                output,
                ravdess_authorizer=authorize_ravdess,
                mayo_authorizer=authorize_mayo,
                producer_sha256=producer,
            )
        finally:
            bridge_core._build_bridge_bundles_at = original_build
            bridge_core.os.write = original_write
        c.eq(calls, {"ravdess": 2, "mayo": 2, "writes": 0})
        c.true(not any(path.name.startswith(".bridge.verify-")
                       for path in output.parent.iterdir()))
        c.true(result["deterministic"])


def test_freeze_stage_apis_are_explicit(c: Check):
    for name in (
        "freeze_bridge_stage",
        "verify_frozen_bridge_stage",
        "initialize_owner_only_key",
    ):
        c.true(hasattr(bridge_core, name), f"missing Task 2 freeze API: {name}")


def test_freeze_stage_is_independent_mode_bound_private_transaction(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        bridge = parent / "bridge"
        run_root = parent / "smoke" / "synthetic-run"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_before = _tree_bytes(bridge)
        result = bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        c.eq(result, {"mode": "smoke", "sample_count": 34, "stage_count": 2})
        inputs = run_root / "inputs"
        expected = {
            "receipts/ravdess.json", "receipts/mayo.json",
            *{
                f"artifacts/{stage}/{name}.json"
                for stage in ("ravdess", "mayo")
                for name in ("manifest", "config", "split", "scaler")
            },
        }
        observed = {
            str(path.relative_to(inputs)) for path in inputs.rglob("*") if path.is_file()
        }
        c.eq(observed, expected)
        c.true(all(stat.S_IMODE(path.stat().st_mode) == 0o600
                   for path in inputs.rglob("*") if path.is_file()))
        for stage in ("ravdess", "mayo"):
            receipt_path = inputs / "receipts" / f"{stage}.json"
            receipt = json.loads(receipt_path.read_text("utf-8"))
            c.eq(receipt["mode"], "smoke")
            c.eq(receipt["producer_sha256"], producer)
            c.eq(receipt["bundle_file_count"], 1)
            c.eq(len(receipt["sample_ids"]), receipt["sample_count"])
            c.true(bool(receipt["receipt_hmac"]))
            text_value = receipt_path.read_text("utf-8")
            c.true(all(value not in text_value for value in (
                str(parent), "source_sha256", "private_key", "patient", "session",
            )), "receipt contains only opaque provenance")
        c.eq(_tree_bytes(bridge), bridge_before,
             "freeze-stage never mutates the shared bundle generation")
        verified = bridge_core.verify_frozen_bridge_stage(
            inputs,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        c.eq(verified["mode"], "smoke")
        c.raises(lambda: bridge_core.verify_frozen_bridge_stage(
            inputs,
            bridge,
            mode="formal",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "a smoke receipt cannot authorize formal mode")
        c.raises(lambda: bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), FileExistsError, "committed inputs are immutable and never overwritten")


def test_frozen_receipts_bind_artifacts_mappings_and_hmac(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        bridge = parent / "bridge"
        run_root = parent / "formal"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="formal",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        inputs = run_root / "inputs"
        for stage, key, width, samples, sources, groups in (
            ("ravdess", ravdess.private_key, 23, 2, 2, 2),
            ("mayo", mayo.private_key, 95, 32, 2, 2),
        ):
            receipt_path = inputs / "receipts" / f"{stage}.json"
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            required = {
                "schema", "stage", "mode", "producer_sha256",
                "upstream_manifest_commitments",
                "upstream_generation_closure_hmac", "sample_ids",
                "source_unit_ids", "group_ids", "cache_integrity_ids",
                "window_starts", "original_mapping_sha256",
                "original_canonical_frame_indices",
                "original_source_frame_indices", "original_timestamps",
                "feature_names_sha256", "adapter_sha256",
                "bundle_file_count", "sample_count", "source_unit_count",
                "unique_group_count", "upstream_cache_count",
                "packet_policy", "overlap_pair_count",
                "covered_canonical_position_count", "exclusion_count",
                "bundle_sha256", "bundle_size_bytes",
                "bridge_generation_sha256", "artifact_core_sha256",
                "canonical_key_identity_sha256", "receipt_hmac",
            }
            c.true(required.issubset(receipt),
                   f"{stage} receipt carries the complete private provenance")
            c.eq(receipt["schema"], "dynamic_landmark_bridge_receipt_v1")
            c.eq(receipt["mode"], "formal")
            c.eq(receipt["sample_count"], samples)
            c.eq(receipt["source_unit_count"], sources)
            c.eq(receipt["unique_group_count"], groups)
            c.eq(len(receipt["source_unit_ids"]), samples)
            c.eq(len(receipt["original_canonical_frame_indices"]), samples)
            c.eq(len(receipt["original_source_frame_indices"]), samples)
            c.eq(len(receipt["original_timestamps"]), samples)
            c.true(all(len(packet) == 4 and all(len(window) == 32 for window in packet)
                       for packet in receipt["original_canonical_frame_indices"]),
                   "original canonical mapping is aligned to every bundle slot")
            unsigned = dict(receipt)
            observed_hmac = unsigned.pop("receipt_hmac")
            expected_hmac = hmac.new(
                key,
                b"dynamic-landmark-bridge-receipt-v1\0" + json.dumps(
                    unsigned, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False,
                ).encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            c.eq(observed_hmac, expected_hmac,
                 "receipt HMAC covers every preceding mode-bound field")
            receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
            for artifact_name in ("manifest", "config", "split", "scaler"):
                artifact_path = (
                    inputs / "artifacts" / stage / f"{artifact_name}.json"
                )
                payload = artifact_path.read_bytes()
                artifact = json.loads(payload)
                c.eq(artifact.pop("bridge_receipt_sha256"), receipt_sha256,
                     f"{stage} {artifact_name} cross-links the exact receipt bytes")
                c.eq(artifact.pop("receipt_hmac"), receipt["receipt_hmac"],
                     f"{stage} {artifact_name} binds the keyed receipt")
                c.eq(
                    receipt["artifact_core_sha256"][artifact_name],
                    hashlib.sha256(json.dumps(
                        artifact, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False,
                    ).encode("ascii")).hexdigest(),
                    f"{stage} {artifact_name} canonical core is bound by the receipt",
                )
                c.true(str(artifact["schema_version"]).endswith("_v2"))
                c.eq(artifact["stage"], stage)
                c.eq(artifact["mode"], "formal")
            scaler = json.loads(
                (inputs / "artifacts" / stage / "scaler.json").read_text("ascii")
            )
            c.eq(len(scaler["mean"]), width)
            c.eq(len(scaler["scale"]), width)
            c.true(all(float(value) > 0.0 for value in scaler["scale"]))


def test_freeze_stage_reauthorizes_and_retains_failed_private_staging(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        bridge = parent / "bridge"
        run_root = parent / "smoke" / "fault"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        calls = 0

        def changed_mayo():
            nonlocal calls
            calls += 1
            if calls == 1:
                return mayo
            changed = dict(vars(mayo))
            changed["generation_closure_hmac"] = "0" * 64
            return SimpleNamespace(**changed)

        observed = _caught(lambda: bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=changed_mayo,
            producer_sha256=producer,
        ))
        c.true(isinstance(observed, RuntimeError))
        c.true("retained" in str(observed) and "indeterminate" in str(observed))
        c.true(isinstance(observed.__cause__, ValueError))
        c.true(not (run_root / "inputs").exists())
        residue = [path for path in run_root.glob(".*")
                   if ".inputs.staging-" in path.name]
        c.eq(len(residue), 1)
        c.eq(stat.S_IMODE(residue[0].stat().st_mode), 0o700)
        _assert_failed(c, lambda: bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), "retained frozen-input residue blocks retry")

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        bridge = parent / "bridge"
        run_root = parent / "smoke" / "cleanup-fault"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        calls = 0
        staging_descriptor = None
        close_injected = False
        original_mkdir = bridge_core._mkdir_private_directory_at
        original_close = bridge_core.os.close

        def changed_mayo():
            nonlocal calls
            calls += 1
            if calls == 1:
                return mayo
            changed = dict(vars(mayo))
            changed["generation_closure_hmac"] = "0" * 64
            return SimpleNamespace(**changed)

        def record_staging_descriptor(parent_fd, name, field):
            nonlocal staging_descriptor
            opened = original_mkdir(parent_fd, name, field)
            if name.startswith(".inputs.staging-"):
                staging_descriptor = opened[0]
            return opened

        def fail_staging_close(descriptor):
            nonlocal close_injected
            original_close(descriptor)
            if descriptor == staging_descriptor and not close_injected:
                close_injected = True
                raise OSError("synthetic frozen staging close failure")

        bridge_core._mkdir_private_directory_at = record_staging_descriptor
        bridge_core.os.close = fail_staging_close
        try:
            observed = _caught(lambda: bridge_core.freeze_bridge_stage(
                run_root,
                bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=changed_mayo,
                producer_sha256=producer,
            ))
        finally:
            bridge_core._mkdir_private_directory_at = original_mkdir
            bridge_core.os.close = original_close
        c.true(close_injected)
        c.true(
            _linear_exception_chain_contains(
                observed, ValueError, "upstream authorization changed",
            )
            and _linear_exception_chain_contains(
                observed, OSError, "synthetic frozen staging close failure",
            ),
            "freeze retained-state error chains primary and cleanup failures",
        )


def _cli_common_args(parent: Path) -> tuple[list[str], Path, Path]:
    ravdess_root = parent / "ravdess-public-source"
    ravdess_key = ravdess_root / ".semantic23_private_id_key"
    mayo_root = parent / "mayo-live-root-privacy-sentinel"
    legacy_root = parent / "mayo-legacy-root-privacy-sentinel"
    cache_root = parent / "deidentified-mayo-cache"
    exposure = parent / "deidentified-mayo-exposure.json"
    mayo_key = parent / "pretraining" / ".mayo_ssl_hmac.key"
    return ([
        "--ravdess-data-root", str(ravdess_root),
        "--ravdess-key", str(ravdess_key),
        "--mayo-data-root", str(mayo_root),
        "--mayo-existing-export-root", str(legacy_root),
        "--mayo-cache-root", str(cache_root),
        "--mayo-exposure-manifest", str(exposure),
        "--mayo-key", str(mayo_key),
    ], mayo_root, legacy_root)


def _captured_cli_call(cli, arguments: list[str]):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = cli.main(arguments)
    return result, stdout.getvalue(), stderr.getvalue()


def _captured_cli_native_call(cli, arguments: list[str]):
    stdout = io.StringIO()
    stderr = io.StringIO()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    observed: object = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as native_stdout, \
                tempfile.TemporaryFile(mode="w+b") as native_stderr:
            os.dup2(native_stdout.fileno(), 1)
            os.dup2(native_stderr.fileno(), 2)
            try:
                with contextlib.redirect_stdout(stdout), \
                        contextlib.redirect_stderr(stderr):
                    observed = _caught(lambda: cli.main(arguments))
            finally:
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
            native_stdout.seek(0)
            native_stderr.seek(0)
            stdout.write(native_stdout.read().decode("utf-8", "replace"))
            stderr.write(native_stderr.read().decode("utf-8", "replace"))
    finally:
        os.close(saved_stdout)
        os.close(saved_stderr)
    return observed, stdout.getvalue(), stderr.getvalue()


def _private_root_representations(path: Path) -> tuple[str, ...]:
    absolute = os.path.abspath(os.fspath(path))
    return tuple(dict.fromkeys((
        absolute,
        path.name,
        os.path.relpath(absolute, os.fspath(ROOT)),
        absolute.encode("utf-8").hex(),
        base64.b64encode(absolute.encode("utf-8")).decode("ascii"),
        base64.urlsafe_b64encode(absolute.encode("utf-8")).decode("ascii"),
    )))


def test_cli_has_exact_subcommands_and_live_mayo_roots_are_preoutput_required(c: Check):
    cli = _load_cli()
    parser = cli._parser()
    subcommands = set()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            subcommands.update(choices)
    c.eq(subcommands, {
        "initialize-mayo-key", "inventory", "build-bundles",
        "freeze-stage", "verify-determinism",
    })

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        cli.PRETRAINING_ROOT = parent / "pretraining"
        common, mayo_root, legacy_root = _cli_common_args(parent)
        target = cli.PRETRAINING_ROOT / "bridge"
        commands = {
            "inventory": [],
            "build-bundles": ["--output-root", str(target)],
            "freeze-stage": [
                "--bridge-root", str(target), "--mode", "smoke",
                "--run-id", "missing-root-check",
            ],
            "verify-determinism": ["--bridge-root", str(target)],
        }
        for command, tail in commands.items():
            full = [command, *common, *tail]
            for missing in ("--mayo-data-root", "--mayo-existing-export-root"):
                index = full.index(missing)
                arguments = full[:index] + full[index + 2:]
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    c.raises(lambda arguments=arguments: cli.main(arguments), SystemExit,
                             f"{command} rejects missing {missing}")
                c.eq(stdout.getvalue(), "", "missing live root emits no stdout")
                emitted_error = stderr.getvalue()
                for private_root in (mayo_root, legacy_root):
                    c.true(all(value not in emitted_error
                               for value in _private_root_representations(private_root)),
                           "missing-root parser stderr contains no supplied root representation")
                c.true(not target.exists())
                c.true(not cli.PRETRAINING_ROOT.exists())

        extra_private_root = parent / "mayo-private-root-sentinel"
        observed, emitted_stdout, emitted_stderr = _captured_cli_native_call(
            cli,
            ["inventory", *common, str(extra_private_root)],
        )
        c.true(isinstance(observed, SystemExit))
        c.eq(emitted_stdout, "", "unknown private path emits no stdout")
        for private_root in (mayo_root, legacy_root, extra_private_root):
            c.true(
                all(
                    value not in emitted_stderr
                    for value in _private_root_representations(private_root)
                ),
                "parser failures never echo a supplied Mayo root",
            )
        c.true(not target.exists() and not cli.PRETRAINING_ROOT.exists())

        key_path = parent / "standalone-key"
        args = parser.parse_args([
            "initialize-mayo-key", "--key-path", str(key_path),
        ])
        c.eq(args.command, "initialize-mayo-key",
             "key initialization is the sole Mayo-root-independent command")
        with contextlib.redirect_stderr(io.StringIO()):
            c.raises(lambda: parser.parse_args([
                "verify-determinism", *common, "--bridge-root", str(target),
                "--output-root", str(parent / "caller-controlled-output"),
            ]), SystemExit, "determinism verification accepts no arbitrary output")

        absent_root = parent / "absent-private-root"
        cli.PRETRAINING_ROOT = absent_root
        cli.CANONICAL_MAYO_KEY = absent_root / ".mayo_ssl_hmac.key"
        c.raises(lambda: cli.main([
            "initialize-mayo-key", "--key-path", str(parent / "wrong-key"),
        ]), ValueError, "wrong key target fails before canonical private output creation")
        c.true(not absent_root.exists(),
               "rejected key location causes no canonical-directory side effect")


def test_all_mayo_cli_commands_capture_native_and_logger_root_leaks(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    for command in (
        "inventory", "build-bundles", "freeze-stage", "verify-determinism",
    ):
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            pretraining = parent / "pretraining"
            pretraining.mkdir(mode=0o700)
            cli.PRETRAINING_ROOT = pretraining
            cli.CANONICAL_MAYO_KEY = pretraining / ".mayo_ssl_hmac.key"
            cli._producer_sha256 = lambda: producer
            common, mayo_root, legacy_root = _cli_common_args(parent)
            representations = (
                *_private_root_representations(mayo_root),
                *_private_root_representations(legacy_root),
            )
            logger = logging.getLogger(f"mayo-root-leak-{command}")
            logger.propagate = False
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            def authorize_ravdess():
                return ravdess

            def authorize_mayo():
                os.write(1, (representations[0] + "\n").encode("utf-8"))
                os.write(2, (representations[1] + "\n").encode("utf-8"))
                os.write(1, (representations[6] + "\n").encode("utf-8"))
                os.write(2, (representations[7] + "\n").encode("utf-8"))
                logger.error("native-path=%s", representations[3])
                logger.error("encoded-path=%s", representations[4])
                logger.error("relative-path=%s", representations[2])
                logger.error("legacy-encoded-path=%s", representations[10])
                return mayo

            setattr(authorize_ravdess, "captured_authorizations", (ravdess,))
            setattr(authorize_mayo, "captured_authorizations", (mayo,))
            cli._authorization_factories = lambda _args: (
                authorize_ravdess, authorize_mayo,
            )
            cli._live_privacy_inventories = lambda _args: (
                SimpleNamespace(member_sha256={"opaque.csv": "7" * 64}),
                SimpleNamespace(),
            )
            cli._scan_private_trees = lambda *_args, **_kwargs: (True, True, 0)

            def fake_build(_output, *, ravdess_authorizer,
                           mayo_authorizer, producer_sha256):
                c.eq(producer_sha256, producer)
                ravdess_authorizer()
                mayo_authorizer()
                return {
                    "ravdess": {"sample_count": 2},
                    "mayo": {"sample_count": 32},
                }

            def fake_freeze(_run, _bridge, *, mode, ravdess_authorizer,
                            mayo_authorizer, producer_sha256):
                c.eq(producer_sha256, producer)
                ravdess_authorizer()
                mayo_authorizer()
                return {"mode": mode, "sample_count": 34, "stage_count": 2}

            def fake_verify(_bridge, *, ravdess_authorizer, mayo_authorizer,
                            producer_sha256, before_authorization,
                            finalize_locked):
                c.eq(producer_sha256, producer)
                before_authorization()
                ravdess_authorizer()
                mayo_authorizer()
                finalize_locked()
                return {
                    "bundle_count": 2,
                    "bundle_total_bytes": 1,
                    "deterministic": True,
                    "size_ok": True,
                }

            cli.build_bridge_bundles = fake_build
            cli.freeze_bridge_stage = fake_freeze
            cli.verify_bridge_generation = fake_verify
            bridge = pretraining / "bridge"
            tails = {
                "inventory": [],
                "build-bundles": ["--output-root", str(bridge)],
                "freeze-stage": [
                    "--bridge-root", str(bridge), "--mode", "smoke",
                    "--run-id", "fd-privacy",
                ],
                "verify-determinism": ["--bridge-root", str(bridge)],
            }
            try:
                observed, stdout, stderr = _captured_cli_native_call(
                    cli, [command, *common, *tails[command]],
                )
            finally:
                logger.removeHandler(handler)
                handler.close()
            c.true(isinstance(observed, ValueError),
                   f"{command} turns captured Mayo-root output into failure")
            c.eq(str(observed), "private Mayo command failed",
                 f"{command} exposes only one generic failure")
            emitted = stdout + stderr + str(observed)
            c.true(all(value not in emitted for value in representations),
                   f"{command} emits no Mayo-root representation")


def test_mayo_cli_capture_starts_before_private_root_resolution(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        mayo_root = parent / "mayo-private-symlink-loop"
        mayo_root.symlink_to(mayo_root)
        args = SimpleNamespace(
            mayo_data_root=mayo_root,
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            observed = _caught(lambda: cli._run_mayo_cli_captured(
                args, lambda: {"ok": True},
            ))
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")
        emitted = stdout.getvalue() + stderr.getvalue() + str(observed)
        c.true(mayo_root.name not in emitted,
               "root-resolution setup failures expose no private basename")


def test_mayo_cli_capture_closes_dup_when_fdopen_construction_fails(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        args = SimpleNamespace(
            mayo_data_root=parent / "mayo-private-live",
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        original_fdopen = cli.os.fdopen
        duplicated_for_stream: list[int] = []

        def fail_fdopen(descriptor, *_positional, **_keywords):
            duplicated_for_stream.append(descriptor)
            raise OSError("synthetic fdopen construction failure")

        cli.os.fdopen = fail_fdopen
        try:
            observed = _caught(lambda: cli._run_mayo_cli_captured(
                args, lambda: {"ok": True},
            ))
        finally:
            cli.os.fdopen = original_fdopen
        still_open: list[int] = []
        for descriptor in duplicated_for_stream:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            still_open.append(descriptor)
            os.close(descriptor)
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")
        c.true(bool(duplicated_for_stream), "fault reaches duplicated text descriptor")
        c.eq(still_open, [], "fdopen construction failure leaks no duplicated FD")


def test_mayo_cli_capture_closes_pipe_when_drain_thread_start_fails(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        args = SimpleNamespace(
            mayo_data_root=parent / "mayo-private-live",
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        original_pipe = cli.os.pipe
        original_thread = cli.threading.Thread
        created_descriptors: list[int] = []

        def tracked_pipe():
            descriptors = original_pipe()
            created_descriptors.extend(descriptors)
            return descriptors

        class StartFailureThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("synthetic drain thread start failure")

        cli.os.pipe = tracked_pipe
        cli.threading.Thread = StartFailureThread
        try:
            observed = _caught(lambda: cli._run_mayo_cli_captured(
                args, lambda: {"ok": True},
            ))
        finally:
            cli.threading.Thread = original_thread
            cli.os.pipe = original_pipe
        still_open: list[int] = []
        for descriptor in created_descriptors:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            still_open.append(descriptor)
            os.close(descriptor)
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")
        c.true(bool(created_descriptors), "fault occurs after pipe creation")
        c.eq(still_open, [], "drain start failure leaks no pipe descriptor")


def test_mayo_cli_native_capture_has_a_hard_storage_bound(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        args = SimpleNamespace(
            mayo_data_root=parent / "mayo-private-live",
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        original_temporary_file = cli.tempfile.TemporaryFile
        tracked: list[object] = []

        class TrackedTemporaryFile:
            def __init__(self, resource):
                self.resource = resource
                self.maximum_size = 0

            @property
            def closed(self):
                return self.resource.closed

            def close(self):
                if not self.resource.closed:
                    self.maximum_size = max(
                        self.maximum_size,
                        int(os.fstat(self.resource.fileno()).st_size),
                    )
                return self.resource.close()

            def __getattr__(self, name):
                return getattr(self.resource, name)

        def tracking_temporary_file(*positional, **keywords):
            wrapper = TrackedTemporaryFile(
                original_temporary_file(*positional, **keywords),
            )
            tracked.append(wrapper)
            return wrapper

        payload = b"x" * (64 * 1024)

        def exceed_native_stdout_bound():
            remaining = cli._MAX_MAYO_CLI_CAPTURE_BYTES + len(payload)
            while remaining:
                chunk = payload[:min(len(payload), remaining)]
                written = os.write(1, chunk)
                c.true(written > 0, "native capture writer makes progress")
                remaining -= written
            return {"ok": True}

        cli.tempfile.TemporaryFile = tracking_temporary_file
        try:
            observed = _caught(lambda: cli._run_mayo_cli_captured(
                args, exceed_native_stdout_bound,
            ))
        finally:
            cli.tempfile.TemporaryFile = original_temporary_file
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")
        c.eq(len(tracked), 2, "stdout and stderr use two bounded capture sinks")
        c.true(all(item.closed for item in tracked),
               "overflow cleanup closes every capture sink")
        c.true(all(
            item.maximum_size <= cli._MAX_MAYO_CLI_CAPTURE_BYTES
            for item in tracked
        ), "no capture storage exceeds the fixed byte bound")


def test_mayo_cli_rejects_private_root_in_operation_result(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        mayo_root = parent / "mayo-private-live"
        legacy_root = parent / "mayo-private-legacy"
        args = SimpleNamespace(
            mayo_data_root=mayo_root,
            mayo_existing_export_root=legacy_root,
        )
        observed = _caught(lambda: cli._run_mayo_cli_captured(
            args,
            lambda: {"unexpected_private_value": str(mayo_root)},
        ))
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")


def test_mayo_cli_result_bytes_are_frozen_before_fd_restore(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        mayo_root = parent / "mayo-private-live"
        args = SimpleNamespace(
            mayo_data_root=mayo_root,
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        mutable_result = {"status": "ok"}
        captured = cli._run_mayo_cli_captured(
            args, lambda: mutable_result,
        )
        mutable_result["late_private_value"] = str(mayo_root)
        c.true(hasattr(captured, "json_line"))
        c.true(str(mayo_root) not in captured.json_line)
        c.eq(json.loads(captured.json_line), {"status": "ok"})


def test_mayo_cli_rejects_json_escaped_unicode_root_result(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        mayo_root = parent / '患者"秘密\\目录'
        args = SimpleNamespace(
            mayo_data_root=mayo_root,
            mayo_existing_export_root=parent / '旧"导出\\目录',
        )
        observed = _caught(lambda: cli._run_mayo_cli_captured(
            args, lambda: {"unexpected_private_value": str(mayo_root)},
        ))
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")


def test_mayo_cli_capture_recovers_fd_restoration_before_generic_failure(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        args = SimpleNamespace(
            mayo_data_root=parent / "mayo-private-live",
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        original_dup2 = cli.os.dup2
        original_close = cli.os.close
        safety_stdout = cli.os.dup(1)
        safety_stderr = cli.os.dup(2)
        destinations: list[int] = []

        def fail_first_stderr_restore(source, destination, *positional,
                                      **keywords):
            destinations.append(destination)
            if destinations == [1, 2, 2]:
                raise OSError("synthetic first stderr restoration failure")
            return original_dup2(source, destination, *positional, **keywords)

        cli.os.dup2 = fail_first_stderr_restore
        try:
            observed = _caught(lambda: cli._run_mayo_cli_captured(
                args, lambda: {"ok": True},
            ))
        finally:
            cli.os.dup2 = original_dup2
            original_dup2(safety_stderr, 2)
            original_dup2(safety_stdout, 1)
            original_close(safety_stderr)
            original_close(safety_stdout)
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")
        c.true(destinations.count(2) >= 3,
               "a failed restoration is retried before descriptor cleanup")


def test_mayo_cli_capture_retries_failed_close_and_attempts_every_fd(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        args = SimpleNamespace(
            mayo_data_root=parent / "mayo-private-live",
            mayo_existing_export_root=parent / "mayo-private-legacy",
        )
        original_dup = cli.os.dup
        original_close = cli.os.close
        duplicated: list[int] = []
        close_calls: list[int] = []

        def tracked_dup(descriptor):
            duplicate = original_dup(descriptor)
            duplicated.append(duplicate)
            return duplicate

        def fail_first_close(descriptor):
            close_calls.append(descriptor)
            if (
                len(duplicated) >= 2
                and descriptor == duplicated[1]
                and close_calls.count(descriptor) == 1
            ):
                raise OSError("synthetic first saved-FD close failure")
            return original_close(descriptor)

        cli.os.dup = tracked_dup
        cli.os.close = fail_first_close
        try:
            observed = _caught(lambda: cli._run_mayo_cli_captured(
                args, lambda: {"ok": True},
            ))
        finally:
            cli.os.close = original_close
            cli.os.dup = original_dup
        still_open: list[int] = []
        for descriptor in duplicated:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            still_open.append(descriptor)
            original_close(descriptor)
        c.true(isinstance(observed, ValueError))
        c.eq(str(observed), "private Mayo command failed")
        c.eq(still_open, [], "all duplicated descriptors are closed after one fault")
        c.true(len(duplicated) >= 4 and close_calls.count(duplicated[1]) >= 2,
               "failed saved-FD close is retried while later closes are attempted")


def test_cli_synthetic_five_command_flow_is_private_and_deterministic(c: Check):
    cli = _load_cli()
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        pretraining = parent / "pretraining"
        pretraining.mkdir(mode=0o700)
        canonical_key = pretraining / ".mayo_ssl_hmac.key"
        cli.PRETRAINING_ROOT = pretraining
        cli.CANONICAL_MAYO_KEY = canonical_key
        cli.authorize_committed_ravdess_semantic23 = (
            lambda *_args, **_kwargs: ravdess
        )
        cli.authorize_committed_mayo_ssl_generation = (
            lambda *_args, **_kwargs: mayo
        )
        synthetic_ravdess_inventory = SimpleNamespace(
            member_sha256={"Actor_01/raw_trial.csv": "7" * 64},
        )
        synthetic_mayo_inventory = SimpleNamespace(
            video_instances=(), long_unique_videos=(), duplicate_videos=(),
            short_videos=(), arkit_trajectories=(), arkit_sessions=(),
            metadata_only_sessions=(),
        )
        cli.audit_ravdess_inventory = lambda *_args, **_kwargs: (
            synthetic_ravdess_inventory
        )
        cli.inventory_mayo_sources = lambda *_args, **_kwargs: (
            synthetic_mayo_inventory
        )
        cli._producer_sha256 = lambda: producer

        outputs: list[str] = []
        result, stdout, stderr = _captured_cli_call(cli, [
            "initialize-mayo-key", "--key-path", str(canonical_key),
        ])
        c.eq(result, 0)
        c.eq(stderr, "")
        outputs.extend((stdout, stderr))
        c.eq(len(canonical_key.read_bytes()), 32)
        c.eq(stat.S_IMODE(canonical_key.stat().st_mode), 0o600)
        first_identity = (canonical_key.stat().st_dev, canonical_key.stat().st_ino)
        first_bytes = canonical_key.read_bytes()
        _captured_cli_call(cli, [
            "initialize-mayo-key", "--key-path", str(canonical_key),
        ])
        c.eq((canonical_key.stat().st_dev, canonical_key.stat().st_ino), first_identity)
        c.eq(canonical_key.read_bytes(), first_bytes,
             "an existing canonical key is validated and never replaced")

        common, mayo_root, legacy_root = _cli_common_args(parent)
        ravdess_key = Path(common[common.index("--ravdess-key") + 1])
        ravdess_key.parent.mkdir(parents=True)
        ravdess_key.write_bytes(ravdess.private_key)
        ravdess_key.chmod(0o600)
        mayo_values = dict(vars(mayo))
        mayo_values["private_key"] = first_bytes
        mayo = SimpleNamespace(**mayo_values)
        bridge = pretraining / "bridge"
        result, stdout, stderr = _captured_cli_call(
            cli, ["inventory", *common],
        )
        c.eq(result, 0)
        c.eq(stderr, "")
        inventory = json.loads(stdout)
        c.eq(inventory, {
            "mayo_recordings": 2,
            "mayo_source_units": 2,
            "ravdess_actors": 2,
            "ravdess_trials": 2,
        })
        outputs.extend((stdout, stderr))

        result, stdout, stderr = _captured_cli_call(cli, [
            "build-bundles", *common, "--output-root", str(bridge),
        ])
        c.eq(result, 0)
        c.eq(stderr, "")
        outputs.extend((stdout, stderr))
        c.eq(json.loads(stdout)["bundle_count"], 2)

        result, stdout, stderr = _captured_cli_call(cli, [
            "freeze-stage", *common, "--bridge-root", str(bridge),
            "--mode", "smoke", "--run-id", "privacy-seed0",
        ])
        c.eq(result, 0)
        c.eq(stderr, "")
        outputs.extend((stdout, stderr))
        run_root = pretraining / "smoke" / "privacy-seed0"
        c.true((run_root / "inputs").is_dir())

        result, stdout, stderr = _captured_cli_call(cli, [
            "verify-determinism", *common, "--bridge-root", str(bridge),
            "--run-root", str(run_root),
        ])
        c.eq(result, 0)
        c.eq(stderr, "")
        outputs.extend((stdout, stderr))
        verification = json.loads(stdout)
        c.eq(set(verification), {
            "bundle_count", "bundle_total_bytes", "deterministic", "modes_ok",
            "non_0600_private_file_count", "privacy_ok", "size_ok",
        })
        c.true(all(verification[name] for name in (
            "deterministic", "modes_ok", "privacy_ok", "size_ok",
        )))

        persisted = b"\n".join(
            path.read_bytes()
            for root in (bridge, run_root)
            for path in root.rglob("*") if path.is_file()
        ).decode("latin1")
        emitted = "".join(outputs)
        for private_root in (mayo_root, legacy_root):
            for representation in _private_root_representations(private_root):
                c.true(representation not in persisted,
                       "private Mayo root representation is absent from artifacts")
                c.true(representation not in emitted,
                       "private Mayo root representation is absent from stdout/stderr")

        leaked = run_root / "inputs" / "artifacts" / "mayo" / "manifest.json"
        leaked.chmod(0o600)
        leaked.write_text(json.dumps({
            "leak": _private_root_representations(mayo_root)[3],
        }), encoding="ascii")
        leaked.chmod(0o600)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            c.raises(lambda: cli.main([
                "verify-determinism", *common, "--bridge-root", str(bridge),
                "--run-root", str(run_root),
            ]), ValueError, "run-root privacy tampering fails verification")
        c.eq(stdout.getvalue(), "", "failed verification emits no aggregate JSON")
        c.true(all(value not in stderr.getvalue()
                   for value in _private_root_representations(mayo_root)))


def test_parent_swap_fails_closed_without_attack_publication(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        safe = root / "safe"
        parked = root / "parked"
        attack = root / "attack"
        safe.mkdir(mode=0o700)
        attack.mkdir()
        output = safe / "bridge"
        swapped = False

        def swap_parent_once():
            nonlocal swapped
            if not swapped:
                safe.rename(parked)
                safe.symlink_to(attack, target_is_directory=True)
                swapped = True
            return ravdess

        c.raises(lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=swap_parent_once,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "build rejects a parent swapped by a live authorizer")
        c.eq(list(attack.iterdir()), [],
             "build never stages or publishes through the attacker symlink")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        safe = root / "safe"
        parked = root / "parked"
        attack = root / "attack"
        safe.mkdir(mode=0o700)
        attack.mkdir()
        swapped = False

        def swap_run_parent_once():
            nonlocal swapped
            if not swapped:
                safe.rename(parked)
                safe.symlink_to(attack, target_is_directory=True)
                swapped = True
            return ravdess

        c.raises(lambda: bridge_core.freeze_bridge_stage(
            safe / "run",
            bridge,
            mode="smoke",
            ravdess_authorizer=swap_run_parent_once,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "freeze rejects a run parent swapped by a live authorizer")
        c.eq(list(attack.iterdir()), [],
             "freeze never creates a run or inputs tree in the attack directory")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        safe = root / "safe"
        parked = root / "parked"
        attack = root / "attack"
        safe.mkdir(mode=0o700)
        attack.mkdir()
        bridge = safe / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        attack_mtime = attack.stat().st_mtime_ns
        swapped = False

        def swap_verify_parent_once():
            nonlocal swapped
            if not swapped:
                safe.rename(parked)
                safe.symlink_to(attack, target_is_directory=True)
                swapped = True
            return ravdess

        c.raises(lambda: bridge_core.verify_bridge_generation(
            bridge,
            ravdess_authorizer=swap_verify_parent_once,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "verify rejects a parent swap before sibling creation")
        c.eq(attack.stat().st_mtime_ns, attack_mtime,
             "verify never creates then removes a sibling through the attack path")


def test_preexisting_private_parent_symlink_is_never_canonicalized_into_trust(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        attack = root / "attack"
        attack.mkdir(mode=0o700)
        linked = root / "linked"
        linked.symlink_to(attack, target_is_directory=True)
        c.raises(lambda: bridge_core.build_bridge_bundles(
            linked / "bridge",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        ), ValueError, "a preexisting output-parent symlink fails before authorization")
        c.eq(list(attack.iterdir()), [],
             "canonicalization never turns an attacker symlink into a trusted parent")


def test_fsync_parent_swaps_never_publish_into_attack_storage(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        safe = root / "safe"
        parked = root / "parked"
        attack = root / "attack"
        safe.mkdir(mode=0o700)
        attack.mkdir(mode=0o700)
        original_fsync = bridge_core._fsync_directory_fd
        swapped = False

        def swap_after_parent_fsync(descriptor: int):
            nonlocal swapped
            original_fsync(descriptor)
            if not swapped and "bridge" in set(os.listdir(descriptor)):
                safe.rename(parked)
                safe.symlink_to(attack, target_is_directory=True)
                swapped = True

        bridge_core._fsync_directory_fd = swap_after_parent_fsync
        try:
            c.raises(lambda: bridge_core.build_bridge_bundles(
                safe / "bridge",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ), ValueError, "build detects a parent swap during final parent fsync")
        finally:
            bridge_core._fsync_directory_fd = original_fsync
        c.eq(list(attack.iterdir()), [], "build fsync fault writes nothing in attack")
        c.true((parked / "bridge").is_dir(),
               "post-rename fsync fault preserves the indeterminate anchored result")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        run_root = root / "smoke" / "fsync-run"
        run_root.parent.mkdir(mode=0o700)
        attack = root / "attack"
        attack.mkdir(mode=0o700)
        parked = root / "parked-run"
        original_fsync = bridge_core._fsync_directory_fd
        swapped = False

        def swap_run_after_inputs_fsync(descriptor: int):
            nonlocal swapped
            original_fsync(descriptor)
            if not swapped and run_root.is_dir() and (run_root / "inputs").is_dir():
                run_root.rename(parked)
                run_root.symlink_to(attack, target_is_directory=True)
                swapped = True

        bridge_core._fsync_directory_fd = swap_run_after_inputs_fsync
        try:
            c.raises(lambda: bridge_core.freeze_bridge_stage(
                run_root,
                bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ), ValueError, "freeze detects a run-root swap during final fsync")
        finally:
            bridge_core._fsync_directory_fd = original_fsync
        c.eq(list(attack.iterdir()), [], "freeze fsync fault writes nothing in attack")
        c.true((parked / "inputs").is_dir(),
               "post-rename freeze fault preserves the anchored inputs result")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        original_fsync = bridge_core._fsync_directory_fd
        calls = 0

        def forbidden_verify_fsync(_descriptor: int):
            nonlocal calls
            calls += 1
            raise AssertionError("read-only verification must not fsync")

        bridge_core._fsync_directory_fd = forbidden_verify_fsync
        try:
            result = bridge_core.verify_bridge_generation(
                bridge,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            )
        finally:
            bridge_core._fsync_directory_fd = original_fsync
        c.eq(calls, 0, "read-only verification performs no directory durability writes")
        c.true(result["deterministic"])


def test_rename_return_fault_preserves_committed_bridge(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    original_publish = bridge_core._atomic_publish_directory_no_replace_at
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary).resolve() / "bridge"

        def publish_then_raise(*args, **kwargs):
            original_publish(*args, **kwargs)
            raise OSError("fault after successful no-replace syscall")

        bridge_core._atomic_publish_directory_no_replace_at = publish_then_raise
        try:
            c.raises(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256="f" * 64,
            ), OSError, "rename success followed by caller fault is indeterminate")
        finally:
            bridge_core._atomic_publish_directory_no_replace_at = original_publish
        c.true(output.is_dir(), "successful rename is never undone by caller-state ambiguity")
        c.raises(lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        ), FileExistsError, "indeterminate committed bridge cannot be retried")


def test_atomic_publish_reports_racing_destination_collision(c: Check):
    class FailingRename:
        def __call__(self, *args):
            bridge_core.ctypes.set_errno(errno.EEXIST)
            return -1

    class FakeLibC:
        pass

    fake_libc = FakeLibC()
    syscall = FailingRename()
    if sys.platform == "darwin":
        fake_libc.renameatx_np = syscall
    else:
        fake_libc.renameat2 = syscall

    original_cdll = bridge_core.ctypes.CDLL
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        (parent / "staging").mkdir(mode=0o700)
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        bridge_core.ctypes.CDLL = lambda *args, **kwargs: fake_libc
        try:
            c.raises(lambda: bridge_core._atomic_publish_directory_no_replace_at(
                descriptor, "staging", "committed",
            ), FileExistsError, "a racing destination collision stays fail-closed")
        finally:
            bridge_core.ctypes.CDLL = original_cdll
            os.close(descriptor)


def test_freeze_rejects_stale_staging_and_preserves_postpublish_results(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        run_root = root / "smoke" / "stale-run"
        run_root.mkdir(parents=True, mode=0o700)
        run_root.parent.chmod(0o700)
        run_root.chmod(0o700)
        stale = run_root / ".inputs.staging-foreign"
        stale.mkdir(mode=0o700)
        marker = stale / "foreign.bin"
        marker.write_bytes(b"foreign-owned")
        marker.chmod(0o600)
        c.raises(lambda: bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "a stale unknown inputs staging tree blocks freeze")
        c.eq(marker.read_bytes(), b"foreign-owned",
             "freeze never removes or mutates unknown staging storage")
        c.true(not (run_root / "inputs").exists())

        stale.rename(run_root / "foreign-preserved")
        original_validate = bridge_core._validate_frozen_inputs_fd
        validations = 0

        def fail_after_publish(root_fd, expected):
            nonlocal validations
            validations += 1
            result = original_validate(root_fd, expected)
            if validations == 3:
                raise OSError("fault after inputs rename")
            return result

        bridge_core._validate_frozen_inputs_fd = fail_after_publish
        try:
            c.raises(lambda: bridge_core.freeze_bridge_stage(
                run_root,
                bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ), OSError, "post-publish frozen-input validation fault is indeterminate")
        finally:
            bridge_core._validate_frozen_inputs_fd = original_validate
        c.true((run_root / "inputs").is_dir(),
               "post-publish freeze fault never deletes committed inputs")
        c.raises(lambda: bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), FileExistsError, "indeterminate frozen inputs cannot be retried")


def test_cli_authorization_wrapper_repr_redacts_private_key(c: Check):
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        key = Path(temporary).resolve() / "key"
        secret = b"q" * 32
        key.write_bytes(secret)
        key.chmod(0o600)
        wrapped = cli._with_key_identity(
            SimpleNamespace(private_key=secret, marker="safe"), key,
        )
        rendered = repr(wrapped)
        c.true(secret.hex() not in rendered and repr(secret) not in rendered,
               "authorization repr never exposes key bytes")
        c.eq(wrapped.marker, "safe")
        ravdess, _mayo = _synthetic_authorizations()
        privacy = cli._authorization_privacy_snapshot(ravdess, "trials")
        c.true(not hasattr(privacy, "trials") and "features" not in repr(privacy),
               "privacy capture retains no trajectory arrays or raw authorization repr")


def test_privacy_token_builder_scales_to_frozen_ravdess_inventory(c: Check):
    cli = _load_cli()
    trials = tuple(
        SimpleNamespace(cache_sha256=f"{index:064x}")
        for index in range(2452)
    )
    members = {
        f"Actor_{index % 24 + 1:02d}/trial_{index:04d}.csv": f"{index + 10000:064x}"
        for index in range(2452)
    }
    mayo_records = tuple(
        SimpleNamespace(cache_sha256=f"{index + 30000:064x}")
        for index in range(48)
    )
    forbidden = cli._build_live_forbidden_tokens(
        mayo_roots=(Path("/private/mayo-live"), Path("/private/mayo-legacy")),
        ravdess_authorization=SimpleNamespace(
            private_key=b"r" * 32, trials=trials,
        ),
        mayo_authorization=SimpleNamespace(
            private_key=b"m" * 32, recordings=mayo_records,
        ),
        ravdess_inventory=SimpleNamespace(member_sha256=members),
        mayo_inventory=SimpleNamespace(
            video_instances=(), long_unique_videos=(), duplicate_videos=(),
            short_videos=(), arkit_trajectories=(), arkit_sessions=(),
            metadata_only_sessions=(),
        ),
    )
    c.true(0 < len(forbidden.tokens) <= 40_000,
           "frozen 2452-trial privacy facts remain within the fixed token cap")
    c.true(sum(len(value) for value in forbidden.tokens) <= 4 * 1024 * 1024,
           "frozen privacy token bytes remain within the fixed memory cap")
    matcher = cli._ByteMatcher(forbidden.tokens)
    _state, leaked = matcher.feed(trials[-1].cache_sha256.upper().encode("ascii"))
    c.true(leaked, "large matcher detects uppercase raw cache SHA without linear scans")


def test_key_initialization_binds_generated_bytes_fd_and_inode(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        key = parent / "key"
        replacement = b"x" * 32
        original_fsync = bridge_core.os.fsync
        replaced = False

        def replace_during_parent_fsync(descriptor: int):
            nonlocal replaced
            original_fsync(descriptor)
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode) and key.exists() and not replaced:
                replaced = True
                os.replace(key, parent / "displaced-key")
                key.write_bytes(replacement)
                key.chmod(0o600)

        bridge_core.os.fsync = replace_during_parent_fsync
        try:
            c.raises(lambda: bridge_core.initialize_owner_only_key(key), ValueError,
                     "an atomic same-size replacement during dir fsync is rejected")
        finally:
            bridge_core.os.fsync = original_fsync
        c.eq(key.read_bytes(), replacement,
             "failure does not mistake or delete the foreign replacement")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        safe = root / "safe"
        parked = root / "parked"
        attack = root / "attack"
        safe.mkdir()
        attack.mkdir()
        key = safe / "key"
        original_fsync = bridge_core.os.fsync
        swapped = False

        def swap_during_parent_fsync(descriptor: int):
            nonlocal swapped
            original_fsync(descriptor)
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode) and key.exists() and not swapped:
                safe.rename(parked)
                safe.symlink_to(attack, target_is_directory=True)
                swapped = True

        bridge_core.os.fsync = swap_during_parent_fsync
        try:
            c.raises(lambda: bridge_core.initialize_owner_only_key(key), ValueError,
                     "key initialization rejects a parent swap during durability fsync")
        finally:
            bridge_core.os.fsync = original_fsync
        c.eq(list(attack.iterdir()), [], "key initialization never writes attack storage")


def test_key_postpublication_fault_allows_validation_only_retry_without_new_secret(
    c: Check,
):
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        key = parent / "key"
        original_read = bridge_core._read_exact_key_at
        faulted = False

        def validate_then_fault(*args, **kwargs):
            nonlocal faulted
            value = original_read(*args, **kwargs)
            if kwargs.get("expected_identity") is not None and not faulted:
                faulted = True
                raise RuntimeError("synthetic postpublication key validation fault")
            return value

        bridge_core._read_exact_key_at = validate_then_fault
        try:
            c.raises(lambda: bridge_core.initialize_owner_only_key(key), RuntimeError,
                     "postpublication key validation fault remains observable")
        finally:
            bridge_core._read_exact_key_at = original_read

        c.true(faulted and key.is_file(),
               "postpublication fault retains the canonical key")
        before = key.stat()
        before_bytes = key.read_bytes()
        c.true(not any(path.name.startswith(".key.staging-")
                       for path in parent.iterdir()),
               "postpublication fault leaves no key staging residue")

        original_urandom = bridge_core.os.urandom
        generated = 0

        def tracked_urandom(size):
            nonlocal generated
            generated += 1
            return original_urandom(size)

        bridge_core.os.urandom = tracked_urandom
        try:
            c.eq(bridge_core.initialize_owner_only_key(key), False,
                 "retry only revalidates the retained canonical key")
        finally:
            bridge_core.os.urandom = original_urandom

        after = key.stat()
        c.eq(generated, 0, "validation-only retry generates no new secret")
        c.eq((after.st_dev, after.st_ino), (before.st_dev, before.st_ino),
             "validation-only retry preserves the canonical key inode")
        c.eq(key.read_bytes(), before_bytes,
             "validation-only retry preserves the canonical key bytes")


def test_verify_requires_the_exact_committed_generation_tree(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    with tempfile.TemporaryDirectory() as temporary:
        bridge = Path(temporary) / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        )
        extra = bridge / "untracked-private.bin"
        extra.write_bytes(b"not part of the committed tree")
        extra.chmod(0o600)
        c.raises(lambda: bridge_core.verify_bridge_generation(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256="f" * 64,
        ), ValueError, "a mode-0600 extra file invalidates exact-tree verification")


def test_exact_tree_requires_owner_only_directories_for_bridge_and_frozen_inputs(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        run = root / "smoke" / "private-modes"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        for directory in (bridge, bridge / "bundles"):
            directory.chmod(0o755)
            try:
                c.raises(lambda: bridge_core.verify_bridge_generation(
                    bridge,
                    ravdess_authorizer=lambda: ravdess,
                    mayo_authorizer=lambda: mayo,
                    producer_sha256=producer,
                ), ValueError, f"mode 0755 bridge directory is rejected: {directory.name}")
            finally:
                directory.chmod(0o700)

        bridge_core.freeze_bridge_stage(
            run,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        inputs = run / "inputs"
        private_directories = (
            inputs,
            inputs / "receipts",
            inputs / "artifacts",
            inputs / "artifacts" / "ravdess",
            inputs / "artifacts" / "mayo",
        )
        for directory in private_directories:
            directory.chmod(0o755)
            try:
                c.raises(lambda: bridge_core.verify_frozen_bridge_stage(
                    inputs,
                    bridge,
                    mode="smoke",
                    ravdess_authorizer=lambda: ravdess,
                    mayo_authorizer=lambda: mayo,
                    producer_sha256=producer,
                ), ValueError, f"mode 0755 frozen directory is rejected: {directory.name}")
            finally:
                directory.chmod(0o700)


def test_exact_tree_snapshot_has_shared_depth_entry_and_byte_budgets(c: Check):
    required = (
        "_MAX_EXACT_TREE_DEPTH",
        "_MAX_EXACT_TREE_ENTRIES",
        "_MAX_EXACT_TREE_TOTAL_BYTES",
    )
    for name in required:
        c.true(hasattr(bridge_core, name), f"exact-tree scanner exposes {name}")

    original_depth = bridge_core._MAX_EXACT_TREE_DEPTH
    original_entries = bridge_core._MAX_EXACT_TREE_ENTRIES
    original_bytes = bridge_core._MAX_EXACT_TREE_TOTAL_BYTES
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "tree"
            root.mkdir(mode=0o700)
            current = root
            for name in ("a", "b", "c"):
                current = current / name
                current.mkdir(mode=0o700)
            bridge_core._MAX_EXACT_TREE_DEPTH = 2
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                c.raises(
                    lambda: bridge_core._snapshot_exact_private_tree_fd(descriptor),
                    ValueError,
                    "exact-tree recursion depth is bounded before deeper traversal",
                )
            finally:
                os.close(descriptor)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "tree"
            root.mkdir(mode=0o700)
            for index in range(3):
                path = root / f"item-{index}.bin"
                path.write_bytes(b"")
                path.chmod(0o600)
            bridge_core._MAX_EXACT_TREE_DEPTH = original_depth
            bridge_core._MAX_EXACT_TREE_ENTRIES = 2
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                c.raises(
                    lambda: bridge_core._snapshot_exact_private_tree_fd(descriptor),
                    ValueError,
                    "exact-tree entry count is shared and bounded",
                )
            finally:
                os.close(descriptor)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "tree"
            root.mkdir(mode=0o700)
            for name in ("first.bin", "second.bin"):
                path = root / name
                path.write_bytes(b"12345678")
                path.chmod(0o600)
            bridge_core._MAX_EXACT_TREE_ENTRIES = original_entries
            bridge_core._MAX_EXACT_TREE_TOTAL_BYTES = 12
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                c.raises(
                    lambda: bridge_core._snapshot_exact_private_tree_fd(descriptor),
                    ValueError,
                    "exact-tree bytes are bounded across all files",
                )
            finally:
                os.close(descriptor)
    finally:
        bridge_core._MAX_EXACT_TREE_DEPTH = original_depth
        bridge_core._MAX_EXACT_TREE_ENTRIES = original_entries
        bridge_core._MAX_EXACT_TREE_TOTAL_BYTES = original_bytes


def test_producer_lineage_includes_feature_adapter_sources(c: Check):
    cli = _load_cli()
    required_producers = {
        ROOT / "src" / "preprocessing" / "semantic_landmarks.py",
        ROOT / "src" / "preprocessing" / "openface68_semantic.py",
        ROOT / "src" / "datasets" / "dynamic_landmark.py",
    }
    c.true(required_producers.issubset(set(cli._PRODUCER_FILES)),
           "producer closure includes every feature adapter implementation")


def test_producer_digest_binds_live_code_and_exact_held_file_snapshots(c: Check):
    cli = _load_cli()
    c.eq(getattr(cli, "_MAX_PRODUCER_FILE_BYTES", None), 4 * 1024 * 1024)
    c.eq(getattr(cli, "_MAX_PRODUCER_TOTAL_BYTES", None), 32 * 1024 * 1024)
    c.true(hasattr(cli, "_PRODUCER_MODULE_NAMES"))

    source_a = "def execute():\n    return 'live-a'\n"
    source_b = "def execute():\n    return 'disk-b'\n"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        path = root / "producer_fixture.py"
        path.write_text(source_a, encoding="utf-8")
        path.chmod(0o600)
        module_name = "_dynamic_landmark_producer_fixture"

        def loaded(source: str) -> ModuleType:
            module = ModuleType(module_name)
            module.__file__ = str(path)
            exec(compile(source, str(path), "exec"), module.__dict__)
            return module

        original_files = cli._PRODUCER_FILES
        original_modules = cli._PRODUCER_MODULE_NAMES
        previous_module = sys.modules.get(module_name)
        cli._PRODUCER_FILES = (path,)
        cli._PRODUCER_MODULE_NAMES = (module_name,)
        try:
            sys.modules[module_name] = loaded(source_a)
            os.replace(path, root / "producer-a.py")
            path.write_text(source_b, encoding="utf-8")
            path.chmod(0o600)
            live_a_disk_b = cli._producer_sha256()

            sys.modules[module_name] = loaded(source_b)
            live_b_disk_b = cli._producer_sha256()
            c.true(
                live_a_disk_b != live_b_disk_b,
                "producer digest distinguishes imported live code A from disk code B",
            )

            sys.modules[module_name] = loaded(source_a)
            original_read = cli.os.read
            mutated = False

            def replace_during_read(descriptor, size):
                nonlocal mutated
                payload = original_read(descriptor, size)
                if not mutated:
                    replacement = root / "producer-replacement.py"
                    replacement.write_text(source_a + "# changed\n", encoding="utf-8")
                    replacement.chmod(0o600)
                    os.replace(replacement, path)
                    mutated = True
                return payload

            cli.os.read = replace_during_read
            try:
                c.raises(
                    cli._producer_sha256,
                    ValueError,
                    "producer file replacement during a held snapshot fails closed",
                )
            finally:
                cli.os.read = original_read
            c.true(mutated)

            oversized = b"x" * (4 * 1024 * 1024 + 1)
            path.write_bytes(oversized)
            path.chmod(0o600)
            reads = 0

            def count_read(descriptor, size):
                nonlocal reads
                reads += 1
                return original_read(descriptor, size)

            cli.os.read = count_read
            try:
                c.raises(
                    cli._producer_sha256,
                    ValueError,
                    "oversized producer source fails from fstat before read",
                )
            finally:
                cli.os.read = original_read
            c.eq(reads, 0)
        finally:
            cli._PRODUCER_FILES = original_files
            cli._PRODUCER_MODULE_NAMES = original_modules
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def test_producer_live_semantics_binds_defaults_closures_globals_and_imports(
    c: Check,
):
    cli = _load_cli()
    source_a = (
        "POLICY = 3\n"
        "def first(value=1, *, window=2):\n"
        "    return value + window + POLICY\n"
        "def second(value=1, *, window=2):\n"
        "    return value - window + POLICY\n"
        "def make(offset):\n"
        "    def closed(value=0):\n"
        "        return value + offset\n"
        "    return closed\n"
        "closed = make(4)\n"
    )
    source_b = (
        "from _producer_semantics_a import first, second, closed\n"
        "handler = first\n"
        "def dispatch(value=0):\n"
        "    return handler(value) + closed(value)\n"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        path_a = root / "producer_semantics_a.py"
        path_b = root / "producer_semantics_b.py"
        path_a.write_text(source_a, encoding="utf-8")
        path_b.write_text(source_b, encoding="utf-8")
        path_a.chmod(0o600)
        path_b.chmod(0o600)
        module_a_name = "_producer_semantics_a"
        module_b_name = "_producer_semantics_b"

        module_a = ModuleType(module_a_name)
        module_a.__file__ = str(path_a)
        exec(compile(source_a, str(path_a), "exec"), module_a.__dict__)
        previous_a = sys.modules.get(module_a_name)
        previous_b = sys.modules.get(module_b_name)
        sys.modules[module_a_name] = module_a
        module_b = ModuleType(module_b_name)
        module_b.__file__ = str(path_b)
        exec(compile(source_b, str(path_b), "exec"), module_b.__dict__)
        sys.modules[module_b_name] = module_b

        original_files = cli._PRODUCER_FILES
        original_modules = cli._PRODUCER_MODULE_NAMES
        cli._PRODUCER_FILES = (path_a, path_b)
        cli._PRODUCER_MODULE_NAMES = (module_a_name, module_b_name)
        try:
            baseline = cli._producer_sha256()

            original_defaults = module_a.first.__defaults__
            module_a.first.__defaults__ = (9,)
            c.true(cli._producer_sha256() != baseline,
                   "producer digest binds positional defaults")
            module_a.first.__defaults__ = original_defaults

            original_kwdefaults = dict(module_a.first.__kwdefaults__)
            module_a.first.__kwdefaults__["window"] = 9
            c.true(cli._producer_sha256() != baseline,
                   "producer digest binds keyword defaults")
            module_a.first.__kwdefaults__.clear()
            module_a.first.__kwdefaults__.update(original_kwdefaults)

            original_policy = module_a.POLICY
            module_a.POLICY = 9
            c.true(cli._producer_sha256() != baseline,
                   "producer digest binds behavior-bearing module globals")
            module_a.POLICY = original_policy

            cell = module_a.closed.__closure__[0]
            original_cell = cell.cell_contents
            cell.cell_contents = 9
            c.true(cli._producer_sha256() != baseline,
                   "producer digest binds closure cell contents")
            cell.cell_contents = original_cell

            original_handler = module_b.handler
            module_b.handler = module_a.second
            c.true(cli._producer_sha256() != baseline,
                   "producer digest binds imported callable dispatch bindings")
            module_b.handler = original_handler
            c.eq(cli._producer_sha256(), baseline,
                 "restored live semantics reproduce the original digest")
        finally:
            cli._PRODUCER_FILES = original_files
            cli._PRODUCER_MODULE_NAMES = original_modules
            if previous_a is None:
                sys.modules.pop(module_a_name, None)
            else:
                sys.modules[module_a_name] = previous_a
            if previous_b is None:
                sys.modules.pop(module_b_name, None)
            else:
                sys.modules[module_b_name] = previous_b


def test_real_producer_digest_executes_without_monkeypatch_and_is_stable(c: Check):
    cli = _load_cli()
    first = cli._producer_sha256()
    second = cli._producer_sha256()
    c.eq(len(first), 64, "real producer digest is a SHA-256 hex string")
    c.true(all(character in "0123456789abcdef" for character in first))
    c.eq(second, first, "real producer closure has a stable live digest")


def test_producer_live_semantics_binds_nested_code_globals(c: Check):
    cli = _load_cli()
    source = (
        "POLICY = 3\n"
        "def outer():\n"
        "    def inner():\n"
        "        return POLICY\n"
        "    return inner()\n"
    )
    module_name = "_nested_live_producer_fixture"
    module = ModuleType(module_name)
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)
    previous = sys.modules.get(module_name)
    original_modules = cli._PRODUCER_MODULE_NAMES
    sys.modules[module_name] = module
    cli._PRODUCER_MODULE_NAMES = (module_name,)
    try:
        baseline = cli._live_producer_semantics_sha256()
        module.POLICY = 9
        c.true(
            cli._live_producer_semantics_sha256() != baseline,
            "nested code-object globals are part of the live behavior closure",
        )
    finally:
        cli._PRODUCER_MODULE_NAMES = original_modules
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_producer_live_semantics_binds_nested_closure_dispatch(c: Check):
    cli = _load_cli()
    external_name = "_nested_closure_dispatch_external"
    producer_name = "_nested_closure_dispatch_producer"
    external = ModuleType(external_name)
    external.handler = lambda: 1
    source = (
        "def factory(imported):\n"
        "    def execute():\n"
        "        def nested():\n"
        "            return imported.handler()\n"
        "        return nested()\n"
        "    return execute\n"
        f"execute = factory(__import__({external_name!r}))\n"
    )
    producer = ModuleType(producer_name)
    previous_external = sys.modules.get(external_name)
    previous_producer = sys.modules.get(producer_name)
    original_modules = cli._PRODUCER_MODULE_NAMES
    sys.modules[external_name] = external
    exec(compile(source, f"<{producer_name}>", "exec"), producer.__dict__)
    sys.modules[producer_name] = producer
    cli._PRODUCER_MODULE_NAMES = (producer_name,)
    try:
        baseline = cli._live_producer_semantics_sha256()
        external.handler = lambda: 2
        c.true(
            cli._live_producer_semantics_sha256() != baseline,
            "nested code binds imported dispatch reached through a closure cell",
        )
    finally:
        cli._PRODUCER_MODULE_NAMES = original_modules
        if previous_external is None:
            sys.modules.pop(external_name, None)
        else:
            sys.modules[external_name] = previous_external
        if previous_producer is None:
            sys.modules.pop(producer_name, None)
        else:
            sys.modules[producer_name] = previous_producer


def test_real_producer_digest_binds_literal_getattr_import_dispatch(c: Check):
    cli = _load_cli()
    missing = object()
    original = getattr(cli.os, "O_CLOEXEC", missing)
    baseline = cli._live_producer_semantics_sha256()
    replacement = 0x40000000 if original is missing else int(original) ^ 0x40000000
    setattr(cli.os, "O_CLOEXEC", replacement)
    try:
        c.true(
            cli._live_producer_semantics_sha256() != baseline,
            "literal getattr on an imported module binds the selected attribute",
        )
    finally:
        if original is missing:
            delattr(cli.os, "O_CLOEXEC")
        else:
            setattr(cli.os, "O_CLOEXEC", original)
    c.eq(
        cli._live_producer_semantics_sha256(),
        baseline,
        "restoring literal getattr dispatch restores producer provenance",
    )


def test_producer_live_semantics_binds_external_class_call_result_dispatch(c: Check):
    cli = _load_cli()
    external_name = "_external_class_dispatch_fixture"
    producer_name = "_external_class_dispatch_producer"
    external = ModuleType(external_name)
    exec(
        compile(
            "class Worker:\n"
            "    def run(self, value=1):\n"
            "        return POLICY + value\n"
            "    @property\n"
            "    def value(self):\n"
            "        return PROPERTY_POLICY\n"
            "def replacement(self):\n"
            "    return 2\n"
            "POLICY = 1\n"
            "PROPERTY_POLICY = 3\n",
            f"<{external_name}>",
            "exec",
        ),
        external.__dict__,
    )
    producer = ModuleType(producer_name)
    previous_external = sys.modules.get(external_name)
    previous_producer = sys.modules.get(producer_name)
    original_modules = cli._PRODUCER_MODULE_NAMES
    sys.modules[external_name] = external
    exec(
        compile(
            f"from {external_name} import Worker\n"
            "RUN = Worker.run\n"
            "PROP = Worker.value\n"
            "def a_shallow():\n"
            "    return RUN, PROP\n"
            "def z_execute():\n"
            "    worker = Worker()\n"
            "    return worker.run(), worker.value\n",
            f"<{producer_name}>",
            "exec",
        ),
        producer.__dict__,
    )
    sys.modules[producer_name] = producer
    cli._PRODUCER_MODULE_NAMES = (producer_name,)
    try:
        baseline = cli._live_producer_semantics_sha256()
        original_run = external.Worker.run
        external.Worker.run = external.replacement
        try:
            c.true(
                cli._live_producer_semantics_sha256() != baseline,
                "constructed external types bind later invoked instance methods",
            )
        finally:
            external.Worker.run = original_run
        c.eq(cli._live_producer_semantics_sha256(), baseline)
        original_defaults = external.Worker.run.__defaults__
        external.Worker.run.__defaults__ = (9,)
        try:
            c.true(
                cli._live_producer_semantics_sha256() != baseline,
                "constructed external methods bind positional defaults",
            )
        finally:
            external.Worker.run.__defaults__ = original_defaults
        external.POLICY = 9
        try:
            c.true(
                cli._live_producer_semantics_sha256() != baseline,
                "constructed external methods bind referenced policy globals",
            )
        finally:
            external.POLICY = 1
        external.PROPERTY_POLICY = 11
        try:
            c.true(
                cli._live_producer_semantics_sha256() != baseline,
                "constructed external properties bind referenced policy globals",
            )
        finally:
            external.PROPERTY_POLICY = 3
        c.eq(cli._live_producer_semantics_sha256(), baseline)
    finally:
        cli._PRODUCER_MODULE_NAMES = original_modules
        if previous_external is None:
            sys.modules.pop(external_name, None)
        else:
            sys.modules[external_name] = previous_external
        if previous_producer is None:
            sys.modules.pop(producer_name, None)
        else:
            sys.modules[producer_name] = previous_producer


def test_producer_live_semantics_distinguishes_globals_from_attributes(c: Check):
    cli = _load_cli()
    source = (
        "UNUSED = object()\n"
        "def execute(record):\n"
        "    return record.UNUSED\n"
    )
    module_name = "_attribute_name_collision_fixture"
    module = ModuleType(module_name)
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)
    previous = sys.modules.get(module_name)
    original_modules = cli._PRODUCER_MODULE_NAMES
    sys.modules[module_name] = module
    cli._PRODUCER_MODULE_NAMES = (module_name,)
    try:
        baseline = cli._live_producer_semantics_sha256()
        module.UNUSED = object()
        c.eq(
            cli._live_producer_semantics_sha256(),
            baseline,
            "LOAD_ATTR names do not become unrelated module-global bindings",
        )
    finally:
        cli._PRODUCER_MODULE_NAMES = original_modules
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_producer_live_semantics_binds_contextual_modules_and_class_descriptors(
    c: Check,
):
    cli = _load_cli()
    helper_name = "_external_dispatch_helper_fixture"
    producer_name = "_contextual_dispatch_producer_fixture"
    helper_source = (
        "def first():\n"
        "    return 1\n"
        "def second():\n"
        "    return 2\n"
    )
    producer_source = (
        f"import {helper_name} as helper\n"
        "class Policy:\n"
        "    VALUE = 3\n"
        "    def read(self):\n"
        "        return self.VALUE\n"
        "def a_first():\n"
        "    return helper.first()\n"
        "def b_second():\n"
        "    return helper.second()\n"
        "def c_policy():\n"
        "    return Policy().read()\n"
    )
    helper = ModuleType(helper_name)
    exec(compile(helper_source, f"<{helper_name}>", "exec"), helper.__dict__)
    producer = ModuleType(producer_name)
    previous_helper = sys.modules.get(helper_name)
    previous_producer = sys.modules.get(producer_name)
    original_modules = cli._PRODUCER_MODULE_NAMES
    sys.modules[helper_name] = helper
    exec(compile(producer_source, f"<{producer_name}>", "exec"), producer.__dict__)
    sys.modules[producer_name] = producer
    cli._PRODUCER_MODULE_NAMES = (producer_name,)
    try:
        baseline = cli._live_producer_semantics_sha256()
        original_second = helper.second
        helper.second = helper.first
        c.true(
            cli._live_producer_semantics_sha256() != baseline,
            "a later imported-module attribute cannot be hidden by an earlier memo ref",
        )
        helper.second = original_second

        producer.Policy.VALUE = 9
        c.true(
            cli._live_producer_semantics_sha256() != baseline,
            "producer class attributes are behavior-bearing",
        )
        producer.Policy.VALUE = 3

        original_read = vars(producer.Policy)["read"]
        producer.Policy.read = staticmethod(original_read)
        c.true(
            cli._live_producer_semantics_sha256() != baseline,
            "ordinary and staticmethod descriptors have distinct semantics",
        )
        producer.Policy.read = original_read
        c.eq(cli._live_producer_semantics_sha256(), baseline)
    finally:
        cli._PRODUCER_MODULE_NAMES = original_modules
        if previous_helper is None:
            sys.modules.pop(helper_name, None)
        else:
            sys.modules[helper_name] = previous_helper
        if previous_producer is None:
            sys.modules.pop(producer_name, None)
        else:
            sys.modules[producer_name] = previous_producer


def test_producer_live_semantic_closure_is_strict_bounded_and_two_pass(c: Check):
    cli = _load_cli()
    cycle: list[object] = []
    cycle.append(cycle)
    first_cycle = cli._LiveSemanticEncoder(("fixture",)).encode(cycle)
    second_cycle = cli._LiveSemanticEncoder(("fixture",)).encode(cycle)
    c.eq(first_cycle, second_cycle,
         "semantic cycles use deterministic ordinals rather than object IDs")
    c.raises(lambda: cli._LiveSemanticEncoder(("fixture",)).encode(
        [None] * (cli._MAX_LIVE_SEMANTIC_CONTAINER + 1)
    ), ValueError, "semantic containers have an exact item bound")
    original_leaf_bytes = cli._MAX_LIVE_SEMANTIC_LEAF_BYTES
    cli._MAX_LIVE_SEMANTIC_LEAF_BYTES = 32
    try:
        c.raises(
            lambda: cli._LiveSemanticEncoder._sort_token(b"x" * 32),
            ValueError,
            "set sort tokens reject oversized byte leaves before framing",
        )
        c.raises(
            lambda: cli._LiveSemanticEncoder._sort_token("\U0001f642" * 9),
            ValueError,
            "set sort tokens reject oversized strings before UTF-8 allocation",
        )
        c.raises(
            lambda: cli._LiveSemanticEncoder._sort_token(
                (b"x" * 14, b"y" * 14),
            ),
            ValueError,
            "set sort tokens bound aggregate container bytes before joining",
        )
    finally:
        cli._MAX_LIVE_SEMANTIC_LEAF_BYTES = original_leaf_bytes

    source = (
        "UNUSED = object()\n"
        "BEHAVIOR = 1\n"
        "def execute():\n"
        "    return BEHAVIOR\n"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        path = root / "strict_producer.py"
        path.write_text(source, encoding="utf-8")
        path.chmod(0o600)
        module_name = "_strict_live_producer_fixture"
        module = ModuleType(module_name)
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)
        previous = sys.modules.get(module_name)
        original_files = cli._PRODUCER_FILES
        original_modules = cli._PRODUCER_MODULE_NAMES
        sys.modules[module_name] = module
        cli._PRODUCER_FILES = (path,)
        cli._PRODUCER_MODULE_NAMES = (module_name,)
        try:
            baseline = cli._producer_sha256()
            module.UNUSED = object()
            c.eq(cli._producer_sha256(), baseline,
                 "unreferenced unsupported objects are outside the behavior closure")
            module.BEHAVIOR = object()
            c.raises(cli._producer_sha256, ValueError,
                     "referenced unsupported behavior fails closed")
            module.BEHAVIOR = 1

            original_live = cli._live_producer_semantics_sha256
            live_calls = 0

            def replace_source_during_final_live_pass():
                nonlocal live_calls
                live_calls += 1
                if live_calls == 2:
                    replacement = root / "strict_producer_replacement.py"
                    replacement.write_text(source + "# changed\n", encoding="utf-8")
                    replacement.chmod(0o600)
                    os.replace(replacement, path)
                return original_live()

            cli._live_producer_semantics_sha256 = replace_source_during_final_live_pass
            try:
                c.raises(cli._producer_sha256, ValueError,
                         "source replacement during final live pass fails closed")
            finally:
                cli._live_producer_semantics_sha256 = original_live
            path.write_text(source, encoding="utf-8")
            path.chmod(0o600)

            observations = iter(("a" * 64, "b" * 64))
            cli._live_producer_semantics_sha256 = lambda: next(observations)
            try:
                c.raises(cli._producer_sha256, ValueError,
                         "between-pass live semantic mutation fails closed")
            finally:
                cli._live_producer_semantics_sha256 = original_live
        finally:
            cli._PRODUCER_FILES = original_files
            cli._PRODUCER_MODULE_NAMES = original_modules
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous


def test_adapter_lineage_binds_runtime_code_and_exact_mayo_order(c: Check):
    original_adapter = bridge_core.clinical23_v2_to_semantic23

    def forged_zero_adapter(values: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(values).shape, dtype=np.float32)

    bridge_core.clinical23_v2_to_semantic23 = forged_zero_adapter
    try:
        c.raises(lambda: _mayo(128), ValueError,
                 "runtime Mayo adapter must equal the explicit np.take mapping")
        ravdess, mayo = _synthetic_authorizations()
        c.raises(lambda: bridge_core._prepare_bridge_generation(
            ravdess, mayo, producer_sha256="f" * 64,
        ), ValueError, "a monkeypatched adapter cannot mint unchanged lineage")
    finally:
        bridge_core.clinical23_v2_to_semantic23 = original_adapter


def test_privacy_scanner_uses_live_secrets_and_scans_compressed_members(c: Check):
    cli = _load_cli()
    ravdess, mayo = _synthetic_authorizations()
    mayo_root = Path("/private/live/Faces_177")
    legacy_root = Path("/private/legacy/MySlate_177")
    raw_ravdess_name = "Actor_01/01-01-01-01-01-01-01.csv"
    raw_ravdess_sha = "71" * 32
    raw_mayo_sha = "82" * 32
    raw_mayo_path = mayo_root / "Faces_177" / "capture.mov"
    ravdess_inventory = SimpleNamespace(
        member_sha256={raw_ravdess_name: raw_ravdess_sha},
    )
    mayo_asset = SimpleNamespace(
        session_path=raw_mayo_path.parent,
        path=raw_mayo_path,
        source_sha256=raw_mayo_sha,
        export_dir=legacy_root / "Faces_177",
    )
    mayo_inventory = SimpleNamespace(
        video_instances=(mayo_asset,),
        arkit_trajectories=(),
        arkit_sessions=(),
        metadata_only_sessions=(),
    )
    c.true(hasattr(cli, "_build_live_forbidden_tokens"),
           "CLI exposes one in-memory-only live privacy token builder")
    if not hasattr(cli, "_build_live_forbidden_tokens"):
        return
    forbidden = cli._build_live_forbidden_tokens(
        mayo_roots=(mayo_root, legacy_root),
        ravdess_authorization=ravdess,
        mayo_authorization=mayo,
        ravdess_inventory=ravdess_inventory,
        mayo_inventory=mayo_inventory,
    )

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary) / "private-tree"
        scan_root.mkdir()
        scan_root.chmod(0o700)
        raw_leak = scan_root / "raw-cache-leak.bin"
        raw_leak.write_bytes(ravdess.trials[0].cache_sha256.encode("ascii"))
        raw_leak.chmod(0o600)
        c.raises(lambda: cli._scan_private_trees(
            [scan_root.resolve()], forbidden=forbidden,
        ), ValueError, "raw cache SHA leakage is rejected despite mode 0600")

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary) / "private-tree"
        scan_root.mkdir()
        scan_root.chmod(0o700)
        checkpoint = scan_root / "checkpoint.pt"
        with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "archive/data.pkl",
                str(mayo_root).encode("utf-8") + b"\0" + mayo.private_key,
            )
        checkpoint.chmod(0o600)
        c.raises(lambda: cli._scan_private_trees(
            [scan_root.resolve()], forbidden=forbidden,
        ), ValueError, "deflated PyTorch ZIP members cannot hide roots or key bytes")

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary) / "private-tree"
        scan_root.mkdir()
        scan_root.chmod(0o700)
        unsafe = scan_root / "unsafe.npz"
        with zipfile.ZipFile(unsafe, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("../escape.npy", b"benign")
        unsafe.chmod(0o600)
        c.raises(lambda: cli._scan_private_trees(
            [scan_root.resolve()], forbidden=forbidden,
        ), ValueError, "unsafe archive member paths fail closed without extraction")


def _privacy_scanner_fixture(cli):
    """Return one bounded synthetic privacy authorization for scanner tests."""
    empty_inventory = SimpleNamespace(member_sha256={"raw.csv": "a" * 64})
    empty_mayo_inventory = SimpleNamespace(
        video_instances=(), long_unique_videos=(), duplicate_videos=(),
        short_videos=(), arkit_trajectories=(), arkit_sessions=(),
        metadata_only_sessions=(),
    )
    ravdess, mayo = _synthetic_authorizations()
    return cli._build_live_forbidden_tokens(
        mayo_roots=(Path("/private/mayo-a"), Path("/private/mayo-b")),
        ravdess_authorization=ravdess,
        mayo_authorization=mayo,
        ravdess_inventory=empty_inventory,
        mayo_inventory=empty_mayo_inventory,
    )


def test_privacy_scanner_rejects_swapped_root_chain(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        safe = root / "safe"
        parked = root / "parked"
        attack = root / "attack"
        safe.mkdir(mode=0o700)
        attack.mkdir(mode=0o700)
        intended = safe / "tree"
        intended.mkdir(mode=0o700)
        (attack / "tree").mkdir(mode=0o700)
        safe.rename(parked)
        safe.symlink_to(attack, target_is_directory=True)
        c.raises(lambda: cli._scan_private_trees(
            [intended], forbidden=forbidden,
        ), ValueError, "scanner rejects a root whose parent became an attack symlink")


def test_privacy_scanner_requires_owner_only_directories(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        loose = scan_root / "loose-directory"
        loose.mkdir(mode=0o755)
        loose.chmod(0o755)
        c.raises(lambda: cli._scan_private_trees(
            [scan_root], forbidden=forbidden,
        ), ValueError, "every directory in a private result tree is owner-only 0700")


def test_key_initialization_is_atomic_concurrent_and_reopen_bound(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        key = parent / "key"
        worker_count = 4
        write_barrier = threading.Barrier(worker_count)
        original_write = bridge_core.os.write
        results: list[bool] = []
        errors: list[BaseException] = []

        def synchronized_write(descriptor, payload):
            try:
                write_barrier.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                pass
            return original_write(descriptor, payload)

        def initialize():
            try:
                results.append(bridge_core.initialize_owner_only_key(key))
            except BaseException as exc:  # record every racing loser outcome
                errors.append(exc)

        bridge_core.os.write = synchronized_write
        try:
            workers = [threading.Thread(target=initialize) for _ in range(worker_count)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5.0)
            c.true(all(not worker.is_alive() for worker in workers),
                   "concurrent key initialization is bounded")
        finally:
            bridge_core.os.write = original_write
        c.eq(errors, [], "a concurrent loser never observes a partial canonical key")
        c.eq(sorted(results), [False, False, False, True],
             "exactly one atomic key publication wins")
        c.eq(len(key.read_bytes()), 32)
        c.eq(stat.S_IMODE(key.stat().st_mode), 0o600)
        c.true(not any(path.name.startswith(".key.staging-")
                       for path in parent.iterdir()),
               "all losing owner-only staging files are removed")
        c.eq(bridge_core.initialize_owner_only_key(key), False,
             "a complete canonical key is an idempotent safe retry")

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        key = parent / "key"
        displaced = parent / "displaced-key"
        replacement = b"foreign-replacement".ljust(32, b"x")
        original_open = bridge_core.os.open
        replaced = False

        def replace_before_final_reopen(path, flags, *args, **kwargs):
            nonlocal replaced
            if (
                path == key.name
                and not flags & os.O_CREAT
                and key.exists()
                and not replaced
            ):
                replaced = True
                key.rename(displaced)
                key.write_bytes(replacement)
                key.chmod(0o600)
            return original_open(path, flags, *args, **kwargs)

        bridge_core.os.open = replace_before_final_reopen
        try:
            c.raises(lambda: bridge_core.initialize_owner_only_key(key), ValueError,
                     "final reopen binds generated bytes and the committed inode")
        finally:
            bridge_core.os.open = original_open
        c.true(replaced, "the regression reaches the post-close final reopen")
        c.eq(key.read_bytes(), replacement,
             "a foreign replacement is detected but never deleted")


def test_key_initialization_retains_prepublication_write_and_sync_failures(c: Check):
    failure_factories = []

    def partial_write_failure():
        original = bridge_core.os.write
        calls = 0

        def fail(descriptor, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(descriptor, memoryview(payload)[:1])
            raise OSError("injected partial key write failure")

        return "partial-write", "write", fail

    def file_fsync_failure():
        original = bridge_core.os.fsync

        def fail(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("injected key file fsync failure")
            return original(descriptor)

        return "file-fsync", "fsync", fail

    def staging_dir_fsync_failure():
        original = bridge_core.os.fsync
        failed = False

        def fail(descriptor):
            nonlocal failed
            info = os.fstat(descriptor)
            if stat.S_ISDIR(info.st_mode) and not failed:
                failed = True
                raise OSError("injected key staging directory fsync failure")
            return original(descriptor)

        return "directory-fsync", "fsync", fail

    failure_factories.extend((
        partial_write_failure,
        file_fsync_failure,
        staging_dir_fsync_failure,
    ))
    for factory in failure_factories:
        label, attribute, injected = factory()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            key = parent / "key"
            original = getattr(bridge_core.os, attribute)
            setattr(bridge_core.os, attribute, injected)
            try:
                observed = _caught(lambda: bridge_core.initialize_owner_only_key(key))
            finally:
                setattr(bridge_core.os, attribute, original)
            c.true(isinstance(observed, RuntimeError), f"{label} is retained")
            c.true("retained" in str(observed) and "indeterminate" in str(observed))
            c.true(isinstance(observed.__cause__, OSError))
            c.true(not key.exists(), f"{label} never exposes a partial canonical key")
            residue = [path for path in parent.iterdir()
                       if path.name.startswith(".key.staging-")]
            c.eq(len(residue), 1, f"{label} retains one auditable staging inode")
            c.eq(stat.S_IMODE(residue[0].stat().st_mode), 0o600)

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        key = parent / "key"
        original_open = bridge_core.os.open
        original_write = bridge_core.os.write
        original_close = bridge_core.os.close
        staging_descriptor = None
        write_calls = 0
        close_injected = False

        def record_staging_open(path, flags, *args, **kwargs):
            nonlocal staging_descriptor
            descriptor = original_open(path, flags, *args, **kwargs)
            if flags & os.O_CREAT and str(path).startswith(".key.staging-"):
                staging_descriptor = descriptor
            return descriptor

        def fail_after_partial_write(descriptor, payload):
            nonlocal write_calls
            write_calls += 1
            if write_calls == 1:
                return original_write(descriptor, memoryview(payload)[:1])
            raise OSError("synthetic key primary write failure")

        def fail_staging_close(descriptor):
            nonlocal close_injected
            original_close(descriptor)
            if descriptor == staging_descriptor and not close_injected:
                close_injected = True
                raise OSError("synthetic key staging close failure")

        bridge_core.os.open = record_staging_open
        bridge_core.os.write = fail_after_partial_write
        bridge_core.os.close = fail_staging_close
        try:
            observed = _caught(
                lambda: bridge_core.initialize_owner_only_key(key)
            )
        finally:
            bridge_core.os.open = original_open
            bridge_core.os.write = original_write
            bridge_core.os.close = original_close
        c.true(close_injected)
        c.true(
            _linear_exception_chain_contains(
                observed, OSError, "synthetic key primary write failure",
            )
            and _linear_exception_chain_contains(
                observed, OSError, "synthetic key staging close failure",
            ),
            "key retained-state error chains primary and cleanup failures",
        )


def test_verify_failure_is_read_only_and_creates_no_sibling(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        output = parent / "bridge"
        bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        ravdess_calls = 0

        def fail_after_sibling_publication():
            nonlocal ravdess_calls
            ravdess_calls += 1
            if ravdess_calls == 2:
                raise ValueError("injected second in-memory authorization failure")
            return ravdess

        c.raises(lambda: bridge_core.verify_bridge_generation(
            output,
            ravdess_authorizer=fail_after_sibling_publication,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "verification surfaces the later authorization failure")
        c.eq(ravdess_calls, 2, "fault occurs during the second in-memory preparation")
        c.true(not any(path.name.startswith(".bridge.verify-")
                       for path in parent.iterdir()),
               "a failed read-only verification never creates a sibling")


def test_adapter_lineage_is_independent_of_filename_and_marshal_metadata(c: Check):
    source = (
        "def adapter(values, scale=2.0):\n"
        "    return values * scale\n"
    )
    changed_source = (
        "def adapter(values, scale=2.0):\n"
        "    return values * scale + 1.0\n"
    )

    def compiled(filename: str, text: str):
        namespace: dict[str, object] = {}
        exec(compile(text, filename, "exec"), namespace)
        return namespace["adapter"]

    sentinel = np.arange(6, dtype=np.float32).reshape(2, 3)
    metadata = {"schema": "adapter-lineage-test-v1"}
    first = bridge_core._adapter_lineage_sha256(
        metadata, compiled("/clone/A/adapter.py", source), sentinel,
    )
    second = bridge_core._adapter_lineage_sha256(
        metadata, compiled("/clone/B/adapter.py", source), sentinel,
    )
    changed = bridge_core._adapter_lineage_sha256(
        metadata, compiled("/clone/A/adapter.py", changed_source), sentinel,
    )
    c.eq(first, second,
         "source-identical adapters do not inherit checkout paths from co_filename")
    c.true(first != changed,
           "path independence still binds the live executable implementation")


def test_open_failures_do_not_leak_descriptors_and_run_root_is_owner_only(c: Check):
    if not Path("/dev/fd").is_dir():
        return

    def descriptor_count() -> int:
        return len(os.listdir("/dev/fd"))

    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        empty_bridge = root / "empty-bridge"
        empty_inputs = root / "empty-inputs"
        empty_bridge.mkdir(mode=0o700)
        empty_inputs.mkdir(mode=0o700)

        baseline = descriptor_count()
        for _ in range(5):
            c.raises(lambda: bridge_core.freeze_bridge_stage(
                root / "missing-a" / "missing-b" / "run",
                empty_bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ), ValueError, "freeze closes its bridge anchor if run-path preparation fails")
        c.eq(descriptor_count(), baseline,
             "freeze open failure paths release every descriptor")

        baseline = descriptor_count()
        for _ in range(5):
            c.raises(lambda: bridge_core.verify_frozen_bridge_stage(
                empty_inputs,
                root / "missing-bridge",
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ), ValueError, "frozen verification closes inputs if bridge open fails")
        c.eq(descriptor_count(), baseline,
             "frozen verification open failures release every descriptor")

        baseline = descriptor_count()
        for _ in range(5):
            c.raises(lambda: bridge_core.verify_bridge_generation(
                root / "missing-generation",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ), ValueError, "bridge verification closes parent if root open fails")
        c.eq(descriptor_count(), baseline,
             "bridge verification open failures release every descriptor")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        run_root = root / "smoke" / "loose-run"
        run_root.mkdir(parents=True, mode=0o700)
        run_root.parent.chmod(0o700)
        run_root.chmod(0o755)
        c.raises(lambda: bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "a preexisting run root must be exact owner/euid mode 0700")
        c.true(not (run_root / "inputs").exists())

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        mode_root = root / "smoke"
        mode_root.mkdir(mode=0o700)
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        baseline = descriptor_count()
        original_fsync_directory = bridge_core._fsync_directory_fd
        injected = OSError("injected run-parent durability fault")
        failed = False

        def fail_first_directory_sync(descriptor):
            nonlocal failed
            if not failed:
                failed = True
                raise injected
            return original_fsync_directory(descriptor)

        bridge_core._fsync_directory_fd = fail_first_directory_sync
        try:
            observed = _caught(lambda: bridge_core.freeze_bridge_stage(
                mode_root / "run",
                bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ))
        finally:
            bridge_core._fsync_directory_fd = original_fsync_directory
        c.true(observed is injected)
        c.eq(descriptor_count(), baseline,
             "run-root descriptor closes when parent durability fails")
        c.eq(stat.S_IMODE((mode_root / "run").stat().st_mode), 0o700)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        outputs = root / "outputs"
        outputs.mkdir(mode=0o755)
        key = outputs / "dynamic_landmark" / "pretraining" / ".mayo_ssl_hmac.key"
        baseline = descriptor_count()
        original_fsync_directory = bridge_core._fsync_directory_fd
        injected = OSError("injected canonical-parent durability fault")
        failed = False

        def fail_first_directory_sync(descriptor):
            nonlocal failed
            if not failed:
                failed = True
                raise injected
            return original_fsync_directory(descriptor)

        bridge_core._fsync_directory_fd = fail_first_directory_sync
        try:
            observed = _caught(lambda: bridge_core.initialize_owner_only_key(key))
        finally:
            bridge_core._fsync_directory_fd = original_fsync_directory
        c.true(observed is injected)
        c.eq(descriptor_count(), baseline,
             "canonical parent descriptor closes when durability fails")
        c.eq(stat.S_IMODE((outputs / "dynamic_landmark").stat().st_mode), 0o700)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return payload.getvalue()


def _zip_with_actual_central_record_count(
    payload: bytes,
    *,
    actual_record_count: int,
) -> bytes:
    eocd = payload.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise AssertionError("ZIP fixture has no ordinary EOCD")
    central_size = int.from_bytes(payload[eocd + 12:eocd + 16], "little")
    central_offset = int.from_bytes(payload[eocd + 16:eocd + 20], "little")
    declared_count = int.from_bytes(payload[eocd + 10:eocd + 12], "little")
    if not 0 < declared_count <= actual_record_count:
        raise AssertionError("ZIP fixture record counts are invalid")
    central = payload[central_offset:central_offset + central_size]
    name_size = int.from_bytes(central[28:30], "little")
    extra_size = int.from_bytes(central[30:32], "little")
    comment_size = int.from_bytes(central[32:34], "little")
    first_size = 46 + name_size + extra_size + comment_size
    expanded = central + central[:first_size] * (
        actual_record_count - declared_count
    )
    forged_eocd = bytearray(payload[eocd:eocd + 22])
    forged_eocd[12:16] = len(expanded).to_bytes(4, "little")
    return payload[:central_offset] + expanded + bytes(forged_eocd)


def _patch_zip_eocd(
    payload: bytes,
    *,
    field_offset: int,
    width: int,
    value: int,
) -> bytes:
    changed = bytearray(payload)
    eocd = changed.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise AssertionError("ZIP fixture has no ordinary EOCD")
    changed[eocd + field_offset:eocd + field_offset + width] = value.to_bytes(
        width, "little"
    )
    return bytes(changed)


def _scan_zip_payload(cli, payload: bytes, forbidden) -> None:
    cli._scan_zip(
        io.BytesIO(payload),
        cli._ByteMatcher(forbidden.tokens),
        depth=0,
        budget=cli._ScanBudget(),
    )


def test_privacy_zip_preflight_counts_actual_central_records_before_zipfile(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    original_zipfile = cli.zipfile.ZipFile
    calls: list[str] = []

    def tracked_zipfile(*args, **kwargs):
        calls.append("ZipFile")
        return original_zipfile(*args, **kwargs)

    payload = _zip_bytes({"tensor.npy": b"benign"})
    forged = tuple(
        _zip_with_actual_central_record_count(
            payload, actual_record_count=count,
        )
        for count in (2, 128)
    )
    cli.zipfile.ZipFile = tracked_zipfile
    try:
        for value in forged:
            c.raises(
                lambda candidate=value: _scan_zip_payload(
                    cli, candidate, forbidden
                ),
                ValueError,
                "actual central records must match the bounded EOCD count",
            )
    finally:
        cli.zipfile.ZipFile = original_zipfile
    c.eq(calls, [], "forged central records are rejected before ZipFile")


def test_privacy_zip_preflight_accepts_sfx_and_rejects_noncanonical_eocd(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    ordinary = _zip_bytes({"archive/data.pkl": b"benign-checkpoint"})
    sfx = b"#!/bin/sh\nexit 0\n" + ordinary
    _scan_zip_payload(cli, sfx, forbidden)

    eocd = ordinary.rfind(b"PK\x05\x06")
    central_size = int.from_bytes(ordinary[eocd + 12:eocd + 16], "little")
    central_offset = int.from_bytes(ordinary[eocd + 16:eocd + 20], "little")
    invalid = (
        _patch_zip_eocd(
            ordinary, field_offset=8, width=2, value=0xFFFF,
        ),
        _patch_zip_eocd(
            ordinary, field_offset=4, width=2, value=1,
        ),
        _patch_zip_eocd(
            ordinary, field_offset=12, width=4, value=central_size + 1,
        ),
        _patch_zip_eocd(
            ordinary, field_offset=16, width=4, value=central_offset + 1,
        ),
    )
    original_zipfile = cli.zipfile.ZipFile
    calls: list[str] = []

    def tracked_zipfile(*args, **kwargs):
        calls.append("ZipFile")
        return original_zipfile(*args, **kwargs)

    cli.zipfile.ZipFile = tracked_zipfile
    try:
        for payload in invalid:
            c.raises(
                lambda candidate=payload: _scan_zip_payload(
                    cli, candidate, forbidden
                ),
                ValueError,
                "ZIP64, multi-disk, and inconsistent EOCD metadata fail closed",
            )
    finally:
        cli.zipfile.ZipFile = original_zipfile
    c.eq(calls, [], "noncanonical EOCD metadata is rejected before ZipFile")


def test_privacy_zipfile_parses_exact_preflight_snapshot_under_live_mutation(
    c: Check,
):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    original_payload = _zip_bytes({"tensor.npy": b"benign-original"})
    changed_payload = _zip_bytes({"changed.npy": b"benign-mutated"})

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        candidate = scan_root / "checkpoint.pt"
        candidate.write_bytes(original_payload)
        candidate.chmod(0o600)
        original_preflight = cli._preflight_zip_central_directory
        original_zipfile = cli.zipfile.ZipFile
        parsed_payloads: list[bytes] = []
        mutated = False

        def mutate_live_inode_after_preflight(handle, *, entry_limit):
            nonlocal mutated
            result = original_preflight(handle, entry_limit=entry_limit)
            if not mutated:
                candidate.write_bytes(changed_payload)
                candidate.chmod(0o600)
                mutated = True
            return result

        def tracked_zipfile(handle, *args, **kwargs):
            position = handle.tell()
            handle.seek(0)
            parsed_payloads.append(handle.read())
            handle.seek(position)
            return original_zipfile(handle, *args, **kwargs)

        cli._preflight_zip_central_directory = mutate_live_inode_after_preflight
        cli.zipfile.ZipFile = tracked_zipfile
        try:
            c.raises(
                lambda: cli._scan_private_trees(
                    [scan_root], forbidden=forbidden,
                ),
                ValueError,
                "outer commitments reject a live archive inode mutation",
            )
        finally:
            cli._preflight_zip_central_directory = original_preflight
            cli.zipfile.ZipFile = original_zipfile
        c.true(mutated, "the live inode changes only after central preflight")
        c.eq(
            parsed_payloads,
            [original_payload],
            "ZipFile parses exactly the immutable bytes accepted by preflight",
        )


def test_privacy_zip_rejects_payload_bearing_directory_entry(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    sensitive = b"/private/mayo-a"

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        candidate = scan_root / "checkpoint.pt"
        candidate.write_bytes(_zip_bytes({"payload/": sensitive * 1024}))
        candidate.chmod(0o600)
        c.raises(
            lambda: cli._scan_private_trees(
                [scan_root], forbidden=forbidden,
            ),
            ValueError,
            "a slash-suffixed ZIP member cannot hide a compressed payload",
        )


def test_privacy_scanner_handles_sfx_nested_archives_and_shared_budgets(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    sensitive = b"/private/mayo-a"

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        sfx = scan_root / "self-extracting-checkpoint.bin"
        sfx.write_bytes(
            b"#!/bin/sh\nexit 0\n" + _zip_bytes({"archive/data.pkl": sensitive})
        )
        sfx.chmod(0o600)
        c.raises(lambda: cli._scan_private_trees(
            [scan_root], forbidden=forbidden,
        ), ValueError,
        "a legal prefixed/SFX ZIP cannot bypass compressed privacy scanning")

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        benign_inner = _zip_bytes({"tensor.npy": b"benign-tensor"})
        nested = scan_root / "nested.pt"
        nested.write_bytes(_zip_bytes({"archive/inner.npz": benign_inner}))
        nested.chmod(0o600)
        c.eq(cli._scan_private_trees([scan_root], forbidden=forbidden),
             (True, True, 0),
             "a legal two-level archive follows a controlled scanner path")

        nested.write_bytes(_zip_bytes({
            "archive/inner.npz": _zip_bytes({"tensor.npy": sensitive}),
        }))
        nested.chmod(0o600)
        c.raises(lambda: cli._scan_private_trees(
            [scan_root], forbidden=forbidden,
        ), ValueError,
        "a sensitive token in a second-level archive is rejected")

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        for index in range(2):
            path = scan_root / f"part-{index}.npz"
            path.write_bytes(_zip_bytes({"tensor.npy": b"12345678"}))
            path.chmod(0o600)
        original_total = cli._MAX_ZIP_TOTAL_BYTES
        cli._MAX_ZIP_TOTAL_BYTES = 12
        try:
            c.raises(lambda: cli._scan_private_trees(
                [scan_root], forbidden=forbidden,
            ), ValueError,
            "expanded-byte budget is shared across every archive in every root")
        finally:
            cli._MAX_ZIP_TOTAL_BYTES = original_total

        original_entries = cli._MAX_ZIP_ENTRIES
        cli._MAX_ZIP_ENTRIES = 1
        try:
            c.raises(lambda: cli._scan_private_trees(
                [scan_root], forbidden=forbidden,
            ), ValueError,
            "archive-entry budget is shared rather than reset for each ZIP")
        finally:
            cli._MAX_ZIP_ENTRIES = original_entries


def test_oversize_nested_archive_candidate_fails_closed(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    inner = _zip_bytes({
        "tensor.npy": b"padding" * 64 + b"/private/mayo-a",
    })
    c.true(len(inner) > 64, "fixture exceeds the injected nested capture bound")
    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        checkpoint = scan_root / "oversize-nested.pt"
        checkpoint.write_bytes(_zip_bytes({"archive/inner.npz": inner}))
        checkpoint.chmod(0o600)
        original_limit = cli._MAX_NESTED_ZIP_BYTES
        cli._MAX_NESTED_ZIP_BYTES = 64
        try:
            c.raises(lambda: cli._scan_private_trees(
                [scan_root], forbidden=forbidden,
            ), ValueError,
            "a nested archive above the capture bound is rejected, never skipped")
        finally:
            cli._MAX_NESTED_ZIP_BYTES = original_limit


def test_privacy_tree_ledger_rejects_late_add_remove_and_replace(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    verify = getattr(cli, "_verify_tree_scan_ledger", None)
    c.true(callable(verify), "the scanner exposes its held-root closure check")

    for scenario in ("add", "remove", "replace"):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            scan_root = parent / "private-tree"
            scan_root.mkdir(mode=0o700)
            target = scan_root / "first.bin"
            target.write_bytes(b"original")
            target.chmod(0o600)
            foreign = parent / "foreign.bin"
            foreign.write_bytes(b"foreign!")
            foreign.chmod(0o600)
            mutated = False

            def mutate_then_verify(root_fd, ledger):
                nonlocal mutated
                if not mutated:
                    mutated = True
                    if scenario == "add":
                        added = scan_root / "late.bin"
                        added.write_bytes(b"late-add")
                        added.chmod(0o600)
                    elif scenario == "remove":
                        target.unlink()
                    else:
                        os.replace(foreign, target)
                return verify(root_fd, ledger)

            cli._verify_tree_scan_ledger = mutate_then_verify
            try:
                c.raises(
                    lambda: cli._scan_private_trees(
                        [scan_root], forbidden=forbidden,
                    ),
                    ValueError,
                    f"a late {scenario} must invalidate the committed tree",
                )
            finally:
                cli._verify_tree_scan_ledger = verify
            c.true(mutated, f"the {scenario} mutation hook ran")


def test_privacy_tree_ledger_rechecks_earlier_files_across_files_and_roots(
    c: Check,
):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)

    def run_case(roots: list[Path], earlier: Path, message: str) -> None:
        original_scan = cli._scan_regular_fd
        calls = 0

        def mutate_earlier_while_scanning_later(*args, **kwargs):
            nonlocal calls
            result = original_scan(*args, **kwargs)
            calls += 1
            if calls == 2:
                earlier.write_bytes(b"changed!")
                earlier.chmod(0o600)
            return result

        cli._scan_regular_fd = mutate_earlier_while_scanning_later
        try:
            c.raises(
                lambda: cli._scan_private_trees(roots, forbidden=forbidden),
                ValueError,
                message,
            )
        finally:
            cli._scan_regular_fd = original_scan
        c.true(calls >= 2, "the mutation happened after an earlier file scan")

    with tempfile.TemporaryDirectory() as temporary:
        scan_root = Path(temporary).resolve() / "private-tree"
        scan_root.mkdir(mode=0o700)
        earlier = scan_root / "a.bin"
        later = scan_root / "z.bin"
        earlier.write_bytes(b"original")
        later.write_bytes(b"later___")
        earlier.chmod(0o600)
        later.chmod(0o600)
        run_case(
            [scan_root], earlier,
            "mutating an earlier file while scanning a later file must fail",
        )

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        first_root = parent / "private-a"
        second_root = parent / "private-b"
        first_root.mkdir(mode=0o700)
        second_root.mkdir(mode=0o700)
        earlier = first_root / "a.bin"
        later = second_root / "z.bin"
        earlier.write_bytes(b"original")
        later.write_bytes(b"later___")
        earlier.chmod(0o600)
        later.chmod(0o600)
        run_case(
            [first_root, second_root], earlier,
            "mutating a file in an earlier root while scanning a later root must fail",
        )


def test_privacy_tree_final_closure_rechecks_earlier_root_during_later_verify(
    c: Check,
):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    original_verify = cli._verify_tree_scan_ledger

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        first_root = parent / "private-a"
        second_root = parent / "private-b"
        first_root.mkdir(mode=0o700)
        second_root.mkdir(mode=0o700)
        earlier = first_root / "a.bin"
        later = second_root / "z.bin"
        earlier.write_bytes(b"original")
        later.write_bytes(b"later___")
        earlier.chmod(0o600)
        later.chmod(0o600)
        verify_calls = 0
        mutated = False

        def mutate_earlier_during_later_verify(root_fd, ledger):
            nonlocal verify_calls, mutated
            verify_calls += 1
            if verify_calls == 2:
                earlier.write_bytes(b"changed!")
                earlier.chmod(0o600)
                mutated = True
            return original_verify(root_fd, ledger)

        cli._verify_tree_scan_ledger = mutate_earlier_during_later_verify
        try:
            c.raises(
                lambda: cli._scan_private_trees(
                    [first_root, second_root], forbidden=forbidden,
                ),
                ValueError,
                "a final closure rejects earlier-root mutation during later verify",
            )
        finally:
            cli._verify_tree_scan_ledger = original_verify
        c.true(mutated, "the mutation occurs only after the earlier root was verified")


def _assert_failed(c: Check, operation, message: str) -> None:
    try:
        operation()
    except BaseException:
        return
    c.true(False, message)


def test_failed_transactions_never_issue_delete_syscalls(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    active = False
    delete_events: list[str] = []

    def observe(event, _args):
        if active and event in {"os.remove", "os.rmdir"}:
            delete_events.append(event)

    sys.addaudithook(observe)

    def run_without_deletes(operation):
        nonlocal active
        active = True
        try:
            return _caught(operation)
        finally:
            active = False

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        build_parent = root / "build"
        build_parent.mkdir(mode=0o700)
        calls = 0

        def drifting_ravdess():
            nonlocal calls
            calls += 1
            if calls == 1:
                return ravdess
            return _namespace_with(ravdess, generation_closure_hmac="0" * 64)

        observed = run_without_deletes(lambda: bridge_core.build_bridge_bundles(
            build_parent / "bridge",
            ravdess_authorizer=drifting_ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ))
        c.true(isinstance(observed, BaseException))
        c.true(any(".bridge.staging-" in path.name for path in build_parent.iterdir()))

        freeze_parent = root / "freeze"
        freeze_parent.mkdir(mode=0o700)
        bridge = freeze_parent / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        calls = 0

        def drifting_mayo():
            nonlocal calls
            calls += 1
            if calls == 1:
                return mayo
            return _namespace_with(mayo, generation_closure_hmac="0" * 64)

        run = freeze_parent / "smoke" / "run"
        observed = run_without_deletes(lambda: bridge_core.freeze_bridge_stage(
            run,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=drifting_mayo,
            producer_sha256=producer,
        ))
        c.true(isinstance(observed, BaseException))
        c.true(any(".inputs.staging-" in path.name for path in run.iterdir()))

        key_parent = root / "key"
        key_parent.mkdir(mode=0o700)
        original_write = bridge_core.os.write

        def fail_key_write(_descriptor, _payload):
            raise OSError("injected key staging write failure")

        bridge_core.os.write = fail_key_write
        try:
            observed = run_without_deletes(
                lambda: bridge_core.initialize_owner_only_key(key_parent / "key")
            )
        finally:
            bridge_core.os.write = original_write
        c.true(isinstance(observed, BaseException))
        c.true(any(path.name.startswith(".key.staging-")
                   for path in key_parent.iterdir()))

        outputs = root / "outputs"
        outputs.mkdir(mode=0o755)
        original_urandom = bridge_core.os.urandom
        bridge_core.os.urandom = lambda _size: (_ for _ in ()).throw(
            OSError("injected canonical parent failure")
        )
        try:
            observed = run_without_deletes(lambda: bridge_core.initialize_owner_only_key(
                outputs / "dynamic_landmark" / "pretraining" / ".mayo_ssl_hmac.key"
            ))
        finally:
            bridge_core.os.urandom = original_urandom
        c.true(isinstance(observed, OSError))
        c.true((outputs / "dynamic_landmark" / "pretraining").is_dir())

        c.eq(delete_events, [],
             "original os.unlink/os.rmdir callables are never invoked by failure handling")


def test_transaction_writers_retain_partial_files_and_verify_is_zero_write(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64

    def inject_partial_write(operation):
        original_write = bridge_core.os.write
        calls = 0

        def fail(descriptor, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_write(descriptor, memoryview(payload)[:1])
            raise OSError("injected partial transaction write")

        bridge_core.os.write = fail
        try:
            _assert_failed(c, operation, "partial transaction write must fail")
        finally:
            bridge_core.os.write = original_write

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        inject_partial_write(lambda: bridge_core.build_bridge_bundles(
            root / "bridge",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ))
        residue = [path for path in root.iterdir() if ".bridge.staging-" in path.name]
        c.eq(len(residue), 1, "build retains its partial private staging tree")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_before = _tree_bytes(bridge)
        inject_partial_write(lambda: bridge_core.freeze_bridge_stage(
            root / "smoke" / "run",
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ))
        residue = [path for path in root.rglob(".*")
                   if ".inputs.staging-" in path.name]
        c.eq(len(residue), 1, "freeze retains its partial private staging tree")
        c.eq(_tree_bytes(bridge), bridge_before)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        original_write = bridge_core.os.write
        writes = 0

        def forbidden_write(*_args, **_kwargs):
            nonlocal writes
            writes += 1
            raise AssertionError("verification attempted to write")

        bridge_core.os.write = forbidden_write
        try:
            result = bridge_core.verify_bridge_generation(
                bridge,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            )
        finally:
            bridge_core.os.write = original_write
        c.eq(writes, 0)
        c.true(result["deterministic"])
        c.true(not any("verify" in path.name or "staging" in path.name
                       for path in root.iterdir()),
               "verify creates no sibling writer state")


def _caught(operation):
    try:
        operation()
    except BaseException as exc:
        return exc
    return None


def _linear_exception_chain_contains(error, exception_type, message):
    current = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, exception_type) and message in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def test_cleanup_attachment_breaks_implicit_primary_cycle(c: Check):
    primary = ValueError("bridge implicit primary")
    observed = None

    def fail_during_cleanup():
        try:
            raise primary
        finally:
            try:
                raise OSError("bridge implicit cleanup")
            except OSError as cleanup_error:
                c.true(cleanup_error.__context__ is primary)
                outcome = bridge_core._attach_cleanup_causes(
                    primary, (cleanup_error,),
                )
                raise outcome.with_traceback(primary.__traceback__)

    try:
        fail_during_cleanup()
    except BaseException as exc:
        observed = exc
    chain_ids: list[int] = []
    current = observed
    while current is not None and len(chain_ids) < 8:
        chain_ids.append(id(current))
        current = current.__cause__ or current.__context__
    c.eq(len(chain_ids), len(set(chain_ids)))
    c.true(
        _linear_exception_chain_contains(
            observed, ValueError, "bridge implicit primary",
        )
        and _linear_exception_chain_contains(
            observed, OSError, "bridge implicit cleanup",
        ),
        "bridge attachment retains one acyclic primary-cleanup chain",
    )


def _fd_count() -> int | None:
    if not Path("/dev/fd").is_dir():
        return None
    return len(os.listdir("/dev/fd"))


def test_private_creation_faults_close_fds_and_retain_created_storage(c: Check):
    directory_faults = ("mkdir", "open", "fchmod", "fstat", "stat")
    for label in directory_faults:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            baseline = _fd_count()
            injected = OSError(f"injected directory {label} fault")
            originals = {
                "mkdir": bridge_core.os.mkdir,
                "open": bridge_core.os.open,
                "fchmod": bridge_core.os.fchmod,
                "fstat": bridge_core.os.fstat,
                "stat": bridge_core.os.stat,
            }
            opened: list[int] = []
            armed = False
            failed = False

            def mkdir(path, *args, **kwargs):
                nonlocal armed, failed
                if path == "created" and label == "mkdir" and not failed:
                    failed = True
                    raise injected
                result = originals["mkdir"](path, *args, **kwargs)
                if path == "created":
                    armed = True
                return result

            def open_file(path, flags, *args, **kwargs):
                nonlocal failed
                if path == "created" and label == "open" and not failed:
                    failed = True
                    raise injected
                descriptor = originals["open"](path, flags, *args, **kwargs)
                if path == "created":
                    opened.append(descriptor)
                return descriptor

            def fchmod(descriptor, mode):
                nonlocal failed
                if label == "fchmod" and armed and not failed:
                    failed = True
                    raise injected
                return originals["fchmod"](descriptor, mode)

            def fstat(descriptor):
                nonlocal failed
                if label == "fstat" and descriptor in opened and not failed:
                    failed = True
                    raise injected
                return originals["fstat"](descriptor)

            def stat_file(path, *args, **kwargs):
                nonlocal failed
                if label == "stat" and path == "created" and armed and not failed:
                    failed = True
                    raise injected
                return originals["stat"](path, *args, **kwargs)

            bridge_core.os.mkdir = mkdir
            bridge_core.os.open = open_file
            bridge_core.os.fchmod = fchmod
            bridge_core.os.fstat = fstat
            bridge_core.os.stat = stat_file
            try:
                observed = _caught(lambda: bridge_core._mkdir_private_directory_at(
                    parent_fd, "created", "fault-injected private directory",
                ))
            finally:
                for attribute, original in originals.items():
                    setattr(bridge_core.os, attribute, original)
            if label == "mkdir":
                c.true(observed is injected, "mkdir failure creates no residue")
                c.true(not (parent / "created").exists())
            else:
                c.true(isinstance(observed, RuntimeError))
                c.true("retained" in str(observed) and "indeterminate" in str(observed))
                c.true(observed.__cause__ is injected)
                c.true((parent / "created").is_dir(),
                       f"{label} retains the created private directory")
            if baseline is not None:
                c.eq(_fd_count(), baseline, f"{label} closes every acquired descriptor")
            for descriptor in opened:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            os.close(parent_fd)

    file_faults = ("open", "fchmod", "fstat", "stat", "write", "fsync")
    for label in file_faults:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            staging = parent / "staging"
            staging.mkdir(mode=0o700)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
            ledger = bridge_core._OwnedTreeLedger.create(os.fstat(staging_fd))
            baseline = _fd_count()
            injected = OSError(f"injected file {label} fault")
            originals = {
                "open": bridge_core.os.open,
                "fchmod": bridge_core.os.fchmod,
                "fstat": bridge_core.os.fstat,
                "stat": bridge_core.os.stat,
                "write": bridge_core.os.write,
                "fsync": bridge_core.os.fsync,
            }
            target_fds: list[int] = []
            failed = False

            def open_file(path, flags, *args, **kwargs):
                nonlocal failed
                if path == "payload.bin" and label == "open" and not failed:
                    failed = True
                    raise injected
                descriptor = originals["open"](path, flags, *args, **kwargs)
                if path == "payload.bin":
                    target_fds.append(descriptor)
                return descriptor

            def fchmod(descriptor, mode):
                nonlocal failed
                if label == "fchmod" and descriptor in target_fds and not failed:
                    failed = True
                    raise injected
                return originals["fchmod"](descriptor, mode)

            def fstat(descriptor):
                nonlocal failed
                if label == "fstat" and descriptor in target_fds and not failed:
                    failed = True
                    raise injected
                return originals["fstat"](descriptor)

            def stat_file(path, *args, **kwargs):
                nonlocal failed
                if label == "stat" and path == "payload.bin" and not failed:
                    failed = True
                    raise injected
                return originals["stat"](path, *args, **kwargs)

            def write(descriptor, payload):
                nonlocal failed
                if label == "write" and descriptor in target_fds and not failed:
                    failed = True
                    raise injected
                return originals["write"](descriptor, payload)

            def fsync(descriptor):
                nonlocal failed
                if label == "fsync" and descriptor in target_fds and not failed:
                    failed = True
                    raise injected
                return originals["fsync"](descriptor)

            bridge_core.os.open = open_file
            bridge_core.os.fchmod = fchmod
            bridge_core.os.fstat = fstat
            bridge_core.os.stat = stat_file
            bridge_core.os.write = write
            bridge_core.os.fsync = fsync

            try:
                observed = _caught(lambda: bridge_core._write_private_file_at(
                    staging_fd,
                    "payload.bin",
                    b"private-payload",
                    ledger=ledger,
                    relative="payload.bin",
                ))
            finally:
                for attribute, original in originals.items():
                    setattr(bridge_core.os, attribute, original)
            c.true(observed is injected, f"file {label} preserves the primary exception")
            c.true(staging.is_dir(), f"file {label} retains transaction storage")
            if baseline is not None:
                c.eq(_fd_count(), baseline,
                     f"file {label} closes every acquired descriptor")
            os.close(staging_fd)
            os.close(parent_fd)


def test_frozen_sequential_acquisition_failures_release_first_descriptor(c: Check):
    baseline = _fd_count()
    if baseline is None:
        return
    tracked: list[int] = []
    original_open_private = bridge_core._open_private_directory_at

    def track_validation_open(*args, **kwargs):
        result = original_open_private(*args, **kwargs)
        if args[1] == "receipts":
            tracked.append(result[0])
        return result

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / "receipts").mkdir(mode=0o700)
        artifact = root / "artifacts"
        artifact.write_bytes(b"not-a-directory")
        artifact.chmod(0o600)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        bridge_core._open_private_directory_at = track_validation_open
        try:
            for _ in range(5):
                c.raises(lambda: bridge_core._validate_frozen_inputs_fd(
                    root_fd, SimpleNamespace(),
                ), ValueError, "second validation acquire fails")
        finally:
            bridge_core._open_private_directory_at = original_open_private
        c.eq(_fd_count(), baseline + 1,
             "validation keeps only the test-owned root descriptor")
        os.close(root_fd)
        for descriptor in tracked:
            try:
                os.close(descriptor)
            except OSError:
                pass

    baseline = _fd_count()
    tracked = []
    original_mkdir_private = bridge_core._mkdir_private_directory_at

    def fail_second_write_acquire(*args, **kwargs):
        if args[1] == "artifacts":
            raise OSError("injected second frozen writer acquire")
        result = original_mkdir_private(*args, **kwargs)
        if args[1] == "receipts":
            tracked.append(result[0])
        return result

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        bridge_core._mkdir_private_directory_at = fail_second_write_acquire
        try:
            for index in range(5):
                staging = parent / f"staging-{index}"
                staging.mkdir(mode=0o700)
                staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
                ledger = bridge_core._OwnedTreeLedger.create(os.fstat(staging_fd))
                try:
                    c.raises(lambda: bridge_core._write_frozen_inputs_fd(
                        staging_fd, SimpleNamespace(), ledger,
                    ), OSError, "second writer acquire fails")
                finally:
                    os.close(staging_fd)
        finally:
            bridge_core._mkdir_private_directory_at = original_mkdir_private
        c.eq(_fd_count(), baseline,
             "writer closes the first directory when its second acquire fails")
        for descriptor in tracked:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_verify_never_fsyncs_or_creates_a_filesystem_sibling(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        output = parent / "bridge"
        bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        original = bridge_core._fsync_directory_fd
        calls = 0

        def forbidden_fsync(_descriptor: int):
            nonlocal calls
            calls += 1
            raise AssertionError("verification attempted a filesystem fsync")

        bridge_core._fsync_directory_fd = forbidden_fsync
        try:
            result = bridge_core.verify_bridge_generation(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            )
        finally:
            bridge_core._fsync_directory_fd = original
        c.eq(calls, 0)
        c.true(result["deterministic"])
        c.true(not any(path.name.startswith(".bridge.verify-")
                       for path in parent.iterdir()),
               "verification creates no filesystem sibling")


def test_unknown_transaction_residue_blocks_build_verify_and_key_before_secrets(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        stale = parent / ".bridge.staging-foreign"
        stale.mkdir(mode=0o700)
        marker = stale / "marker.bin"
        marker.write_bytes(b"foreign-build")
        marker.chmod(0o600)
        calls = 0

        def authorize():
            nonlocal calls
            calls += 1
            return ravdess

        _assert_failed(c, lambda: bridge_core.build_bridge_bundles(
            parent / "bridge",
            ravdess_authorizer=authorize,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), "unknown bridge staging blocks build")
        c.eq(calls, 0, "build rejects residue before authorization")
        c.eq(marker.read_bytes(), b"foreign-build",
             "build never deletes an unknown inode")
        c.true(not (parent / "bridge").exists())

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        output = parent / "bridge"
        bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        stale = parent / ".bridge.verify-foreign"
        stale.mkdir(mode=0o700)
        marker = stale / "marker.bin"
        marker.write_bytes(b"foreign-verify")
        marker.chmod(0o600)
        calls = 0

        def authorize():
            nonlocal calls
            calls += 1
            return ravdess

        _assert_failed(c, lambda: bridge_core.verify_bridge_generation(
            output,
            ravdess_authorizer=authorize,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), "unknown verifier residue blocks verify")
        c.eq(calls, 0, "verify rejects residue before authorization")
        c.eq(marker.read_bytes(), b"foreign-verify",
             "verify never deletes an unknown inode")

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        stale = parent / ".key.staging-foreign"
        stale.write_bytes(b"foreign-key-stage".ljust(32, b"x"))
        stale.chmod(0o600)
        key = parent / "key"
        original_urandom = bridge_core.os.urandom
        generated = 0

        def urandom(size):
            nonlocal generated
            generated += 1
            return original_urandom(size)

        bridge_core.os.urandom = urandom
        try:
            _assert_failed(c, lambda: bridge_core.initialize_owner_only_key(key),
                           "unknown key staging blocks initialization")
        finally:
            bridge_core.os.urandom = original_urandom
        c.eq(generated, 0, "key residue is rejected before secret generation")
        c.eq(stale.read_bytes(), b"foreign-key-stage".ljust(32, b"x"))
        c.true(not key.exists())


def test_canonical_key_initializer_creates_and_retains_exact_private_parent_chain(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        outputs = root / "outputs"
        outputs.mkdir(mode=0o755)
        outputs.chmod(0o755)
        pretraining = outputs / "dynamic_landmark" / "pretraining"
        key = pretraining / ".mayo_ssl_hmac.key"
        c.eq(bridge_core.initialize_owner_only_key(key), True,
             "canonical initializer creates its fixed missing private chain")
        for directory in (pretraining.parent, pretraining):
            info = directory.stat()
            c.true(stat.S_ISDIR(info.st_mode))
            c.eq(info.st_uid, os.geteuid())
            c.eq(stat.S_IMODE(info.st_mode), 0o700)
            c.true(not directory.is_symlink())
        c.eq(stat.S_IMODE(key.stat().st_mode), 0o600)
        c.eq(len(key.read_bytes()), 32)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        outputs = root / "outputs"
        outputs.mkdir(mode=0o755)
        outputs.chmod(0o755)
        key = outputs / "dynamic_landmark" / "pretraining" / ".mayo_ssl_hmac.key"
        injected = OSError("injected canonical key generation failure")
        original_urandom = bridge_core.os.urandom
        reached = False

        def fail_generation(_size):
            nonlocal reached
            reached = True
            raise injected

        bridge_core.os.urandom = fail_generation
        try:
            observed = _caught(lambda: bridge_core.initialize_owner_only_key(key))
        finally:
            bridge_core.os.urandom = original_urandom
        c.true(reached, "failure is injected only after private parents exist")
        c.true(observed is injected, "canonical parent creation preserves primary error")
        for directory in (
            outputs / "dynamic_landmark",
            outputs / "dynamic_landmark" / "pretraining",
        ):
            c.true(directory.is_dir(), "expected private parents are retained")
            c.eq(stat.S_IMODE(directory.stat().st_mode), 0o700)
        c.eq(stat.S_IMODE(outputs.stat().st_mode), 0o755,
             "preexisting outputs namespace is never changed")


def test_private_transaction_parents_must_be_current_owner_mode_0700(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        outputs = root / "outputs"
        dynamic = outputs / "dynamic_landmark"
        pretraining = dynamic / "pretraining"
        outputs.mkdir(mode=0o755)
        dynamic.mkdir(mode=0o700)
        pretraining.mkdir(mode=0o700)
        pretraining.chmod(0o777)
        key = pretraining / ".mayo_ssl_hmac.key"
        c.raises(
            lambda: bridge_core.initialize_owner_only_key(key),
            ValueError,
            "a pre-existing world-writable canonical key parent is rejected",
        )
        c.true(not key.exists())
        c.true(not any(path.name.startswith("..mayo_ssl_hmac.key.staging-")
                       for path in pretraining.iterdir()))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        pretraining = root / "pretraining"
        pretraining.mkdir(mode=0o700)
        pretraining.chmod(0o777)
        authorization_calls = 0

        def authorize_ravdess():
            nonlocal authorization_calls
            authorization_calls += 1
            return ravdess

        c.raises(
            lambda: bridge_core.build_bridge_bundles(
                pretraining / "bridge",
                ravdess_authorizer=authorize_ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ),
            ValueError,
            "a world-writable bridge transaction parent is rejected",
        )
        c.eq(authorization_calls, 0, "unsafe parent fails before authorization")
        c.true(not any("staging" in path.name for path in pretraining.iterdir()))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        pretraining = root / "pretraining"
        pretraining.mkdir(mode=0o700)
        bridge = pretraining / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        smoke = pretraining / "smoke"
        smoke.mkdir(mode=0o700)
        smoke.chmod(0o777)
        c.raises(
            lambda: bridge_core.freeze_bridge_stage(
                smoke / "run",
                bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ),
            ValueError,
            "a world-writable frozen-input transaction parent is rejected",
        )
        c.true(not (smoke / "run").exists())


def _namespace_with(value, **changes):
    fields = dict(vars(value))
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_public_semantic_gate_uses_original_frozen_counts_before_staging(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    synthetic_contract = {
        name: getattr(bridge_core, name)
        for name in _PRODUCTION_BRIDGE_CONTRACT
    }
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_before = _tree_bytes(bridge)
        _set_bridge_contract(_PRODUCTION_BRIDGE_CONTRACT)
        try:
            operations = (
                ("build", lambda: bridge_core.build_bridge_bundles(
                    root / "second-bridge",
                    ravdess_authorizer=lambda: ravdess,
                    mayo_authorizer=lambda: mayo,
                    producer_sha256=producer,
                )),
                ("freeze", lambda: bridge_core.freeze_bridge_stage(
                    root / "smoke" / "production-gate",
                    bridge,
                    mode="smoke",
                    ravdess_authorizer=lambda: ravdess,
                    mayo_authorizer=lambda: mayo,
                    producer_sha256=producer,
                )),
                ("verify", lambda: bridge_core.verify_bridge_generation(
                    bridge,
                    ravdess_authorizer=lambda: ravdess,
                    mayo_authorizer=lambda: mayo,
                    producer_sha256=producer,
                )),
            )
            for label, operation in operations:
                observed = _caught(operation)
                c.true(isinstance(observed, ValueError),
                       f"{label} rejects jointly shrunken observed/expected counts")
                c.true(not any(
                    ".staging-" in path.name or ".verify-" in path.name
                    for path in root.rglob("*")
                ), f"{label} rejects before creating transaction storage")
            c.true(not (root / "second-bridge").exists())
            c.true(not (root / "smoke" / "production-gate" / "inputs").exists())
            c.eq(_tree_bytes(bridge), bridge_before)
        finally:
            _set_bridge_contract(synthetic_contract)


def test_public_semantic_gate_requires_canonical_ids_and_exact_mayo_v3(c: Check):
    producer = "f" * 64

    def invalid_ravdess_id(field: str):
        ravdess, mayo = _synthetic_authorizations()
        trials = list(ravdess.trials)
        trials[0] = _namespace_with(trials[0], **{field: "not-canonical"})
        return _namespace_with(ravdess, trials=tuple(trials)), mayo

    def invalid_mayo_id(field: str):
        ravdess, mayo = _synthetic_authorizations()
        recordings = list(mayo.recordings)
        recordings[0] = _namespace_with(
            recordings[0], **{field: "not-canonical"},
        )
        return ravdess, _namespace_with(mayo, recordings=tuple(recordings))

    def invalid_commitment(**changes):
        ravdess, mayo = _synthetic_authorizations()
        commitment = dict(mayo.commitment)
        commitment.update(changes)
        return ravdess, _namespace_with(mayo, commitment=commitment)

    def commitment_with_extra_field():
        return invalid_commitment(unexpected_field="must-fail")

    def jointly_shrunken_ravdess():
        ravdess, mayo = _synthetic_authorizations()
        trial = ravdess.trials[0]
        return _namespace_with(
            ravdess,
            trial_count=1,
            actor_count=1,
            source_frames=len(trial.features),
            expected_trial_count=1,
            expected_actor_count=1,
            trials=(trial,),
        ), mayo

    cases = (
        ("RAVDESS trial ID", lambda: invalid_ravdess_id("trial_id")),
        ("RAVDESS actor ID", lambda: invalid_ravdess_id("actor_id")),
        ("RAVDESS cache ID", lambda: invalid_ravdess_id("cache_integrity_id")),
        ("Mayo recording ID", lambda: invalid_mayo_id("recording_id")),
        ("Mayo group ID", lambda: invalid_mayo_id("group_id")),
        ("Mayo cache ID", lambda: invalid_mayo_id("cache_integrity_id")),
        ("jointly shrunken RAVDESS", jointly_shrunken_ravdess),
        ("Mayo v3 exact field set", commitment_with_extra_field),
        ("Mayo v3 lowercase digest", lambda: invalid_commitment(
            cache_tree_aggregate_sha256="G" * 64,
        )),
        ("Mayo v3 48/8/56-equivalent counts", lambda: invalid_commitment(
            mediapipe_file_count=1,
            cache_file_count=2,
        )),
        ("Mayo v3 authorization digest binding", lambda: invalid_commitment(
            collection_manifest_sha256="0" * 64,
        )),
        ("Mayo v3 classification commitment", lambda: invalid_commitment(
            exposure_classification_integrity_id="agg_invalid",
        )),
    )
    for label, factory in cases:
        ravdess, mayo = factory()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "bridge"
            observed = _caught(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ))
            c.true(isinstance(observed, ValueError), f"{label} is rejected")
            c.true(not output.exists(), f"{label} never publishes")
            c.true(not any("staging" in path.name for path in root.iterdir()),
                   f"{label} rejects before staging")


def _overwrite_private_file_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("test overwrite made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def test_build_rejects_postpublish_ravdess_inode_content_mutation(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    original_read = bridge_core._read_private_file_at
    ravdess_reads = 0
    mutated = False
    same_inode = False

    def mutate_after_final_ravdess_read(directory_fd, name, field):
        nonlocal ravdess_reads, mutated, same_inode
        payload = original_read(directory_fd, name, field)
        if name == "ravdess_bundle.npz":
            ravdess_reads += 1
            if ravdess_reads == 3:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                changed = bytes([payload[0] ^ 0x01]) + payload[1:]
                _overwrite_private_file_at(directory_fd, name, changed)
                after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                same_inode = bridge_core._inode_identity(before) == bridge_core._inode_identity(after)
                mutated = True
        return payload

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary).resolve() / "bridge"
        bridge_core._read_private_file_at = mutate_after_final_ravdess_read
        try:
            observed = _caught(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ))
        finally:
            bridge_core._read_private_file_at = original_read
        c.true(mutated and same_inode,
               "fault changes the original RAVDESS inode after its final read")
        c.true(isinstance(observed, ValueError),
               "build cannot return success for a post-read inconsistent generation")
        c.true(output.is_dir(),
               "an inconsistent post-publish generation is retained fail-closed")
        c.raises(lambda: bridge_core.build_bridge_bundles(
            output,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), FileExistsError, "retained canonical generation cannot be retried in place")


def test_frozen_validator_rechecks_an_earlier_receipt_after_later_reads(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        run = root / "smoke" / "closure"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_core.freeze_bridge_stage(
            run,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        generation = bridge_core._prepare_bridge_generation(
            ravdess, mayo, producer_sha256=producer,
        )
        expected = bridge_core._prepare_frozen_inputs(generation, mode="smoke")
        original_read = bridge_core._read_private_file_at
        mutated = False

        def mutate_ravdess_receipt_after_mayo_read(directory_fd, name, field):
            nonlocal mutated
            payload = original_read(directory_fd, name, field)
            if name == "mayo.json" and not mutated:
                ravdess_payload = original_read(
                    directory_fd, "ravdess.json", "earlier RAVDESS receipt",
                )
                changed = bytes([ravdess_payload[0] ^ 0x01]) + ravdess_payload[1:]
                _overwrite_private_file_at(directory_fd, "ravdess.json", changed)
                mutated = True
            return payload

        bridge_core._read_private_file_at = mutate_ravdess_receipt_after_mayo_read
        try:
            observed = _caught(lambda: bridge_core._validate_frozen_inputs_tree(
                run / "inputs", expected,
            ))
        finally:
            bridge_core._read_private_file_at = original_read
        c.true(mutated, "fault changes an earlier receipt after its single-file scan")
        c.true(isinstance(observed, ValueError),
               "frozen validation closes over every earlier file")


def test_generation_validator_rechecks_late_root_and_bundle_directory_changes(c: Check):
    producer = "f" * 64
    for scenario in ("root-add", "root-remove", "bundles-replace"):
        ravdess, mayo = _synthetic_authorizations()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "bridge"
            bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            )
            expected = bridge_core._prepare_bridge_generation(
                ravdess, mayo, producer_sha256=producer,
            )
            root_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            original_read = bridge_core._read_private_file_at
            mutated = False

            def mutate_after_ravdess(directory_fd, name, field):
                nonlocal mutated
                payload = original_read(directory_fd, name, field)
                if name == "ravdess_bundle.npz" and not mutated:
                    if scenario == "root-add":
                        extra = os.open(
                            "late.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600, dir_fd=root_fd,
                        )
                        os.close(extra)
                    elif scenario == "root-remove":
                        os.unlink("bundle_generation.json", dir_fd=root_fd)
                    else:
                        os.rename(
                            "bundles", "parked-bundles",
                            src_dir_fd=root_fd, dst_dir_fd=root_fd,
                        )
                        os.mkdir("bundles", 0o700, dir_fd=root_fd)
                    mutated = True
                return payload

            bridge_core._read_private_file_at = mutate_after_ravdess
            try:
                observed = _caught(lambda: bridge_core._validate_generation_fd(
                    root_fd, expected,
                ))
            finally:
                bridge_core._read_private_file_at = original_read
                os.close(root_fd)
            c.true(mutated, f"{scenario} occurs after the initial exact-name scan")
            c.true(isinstance(observed, ValueError),
                   f"generation closure rejects late {scenario}")


def test_frozen_validator_rechecks_late_receipts_and_artifacts_changes(c: Check):
    producer = "f" * 64
    for scenario in ("receipts-add", "receipts-remove", "artifacts-replace"):
        ravdess, mayo = _synthetic_authorizations()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bridge = root / "bridge"
            run = root / "smoke" / scenario
            bridge_core.build_bridge_bundles(
                bridge,
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            )
            bridge_core.freeze_bridge_stage(
                run,
                bridge,
                mode="smoke",
                ravdess_authorizer=lambda: ravdess,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            )
            generation = bridge_core._prepare_bridge_generation(
                ravdess, mayo, producer_sha256=producer,
            )
            expected = bridge_core._prepare_frozen_inputs(generation, mode="smoke")
            inputs = run / "inputs"
            root_fd = os.open(inputs, os.O_RDONLY | os.O_DIRECTORY)
            original_read = bridge_core._read_private_file_at
            mutated = False

            def mutate_late(directory_fd, name, field):
                nonlocal mutated
                payload = original_read(directory_fd, name, field)
                if scenario.startswith("receipts") and name == "mayo.json" and not mutated:
                    if scenario == "receipts-add":
                        extra = os.open(
                            "late.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600, dir_fd=directory_fd,
                        )
                        os.close(extra)
                    else:
                        os.unlink("ravdess.json", dir_fd=directory_fd)
                    mutated = True
                elif scenario == "artifacts-replace" and name == "manifest.json" and not mutated:
                    os.rename(
                        "artifacts", "parked-artifacts",
                        src_dir_fd=root_fd, dst_dir_fd=root_fd,
                    )
                    os.mkdir("artifacts", 0o700, dir_fd=root_fd)
                    mutated = True
                return payload

            bridge_core._read_private_file_at = mutate_late
            try:
                observed = _caught(lambda: bridge_core._validate_frozen_inputs_fd(
                    root_fd, expected,
                ))
            finally:
                bridge_core._read_private_file_at = original_read
                os.close(root_fd)
            c.true(mutated, f"{scenario} occurs after the initial exact-name scan")
            c.true(isinstance(observed, ValueError),
                   f"frozen closure rejects late {scenario}")


def test_multifd_cleanup_attempts_all_closes_and_preserves_primary_error(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64

    def chain_contains(error: BaseException | None, text: str) -> bool:
        seen: set[int] = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            if text in str(error):
                return True
            error = error.__cause__ or error.__context__
        return False

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        run = root / "smoke" / "close-validation"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_core.freeze_bridge_stage(
            run,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        expected = bridge_core._prepare_frozen_inputs(
            bridge_core._prepare_bridge_generation(
                ravdess, mayo, producer_sha256=producer,
            ),
            mode="smoke",
        )
        root_fd = os.open(run / "inputs", os.O_RDONLY | os.O_DIRECTORY)
        original_open = bridge_core._open_private_directory_at
        original_read = bridge_core._read_private_file_at
        original_close = bridge_core.os.close
        held: dict[str, int] = {}
        close_attempts: list[int] = []
        injected = False
        active = False

        def tracked_open(parent_fd, name, field, **kwargs):
            nonlocal active
            descriptor, info = original_open(parent_fd, name, field, **kwargs)
            if field in {"frozen receipt directory", "frozen artifact directory"}:
                held[field] = descriptor
            if field == "frozen artifact directory":
                active = True
            return descriptor, info

        def fail_read(*_args, **_kwargs):
            raise RuntimeError("primary frozen validation failure")

        def tracked_close(descriptor):
            nonlocal injected
            if active:
                close_attempts.append(descriptor)
            original_close(descriptor)
            if (
                active
                and descriptor == held.get("frozen artifact directory")
                and not injected
            ):
                injected = True
                raise OSError("synthetic frozen validation close failure")

        bridge_core._open_private_directory_at = tracked_open
        bridge_core._read_private_file_at = fail_read
        bridge_core.os.close = tracked_close
        caught: BaseException | None = None
        try:
            try:
                bridge_core._validate_frozen_inputs_fd(root_fd, expected)
            except BaseException as exc:  # noqa: BLE001 - inspect cleanup chain
                caught = exc
        finally:
            bridge_core.os.close = original_close
            bridge_core._read_private_file_at = original_read
            bridge_core._open_private_directory_at = original_open
            original_close(root_fd)
            for descriptor in held.values():
                if descriptor not in close_attempts:
                    try:
                        original_close(descriptor)
                    except OSError:
                        pass
        c.true(injected, "frozen validation close failure was injected")
        c.true(set(held.values()).issubset(set(close_attempts)),
               "frozen validation attempts every held descriptor close")
        c.true(chain_contains(caught, "primary frozen validation failure"),
               "frozen validation cleanup preserves its primary failure")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        staging = root / "inputs-staging"
        staging.mkdir(mode=0o700)
        staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        expected = bridge_core._prepare_frozen_inputs(
            bridge_core._prepare_bridge_generation(
                ravdess, mayo, producer_sha256=producer,
            ),
            mode="smoke",
        )
        ledger = bridge_core._OwnedTreeLedger.create(os.fstat(staging_fd))
        original_mkdir = bridge_core._mkdir_private_directory_at
        original_write = bridge_core._write_private_file_at
        original_close = bridge_core.os.close
        held: dict[str, int] = {}
        close_attempts: list[int] = []
        injected = False
        active = False

        def tracked_mkdir(parent_fd, name, field):
            nonlocal active
            descriptor, info = original_mkdir(parent_fd, name, field)
            if field in {"frozen receipt directory", "frozen artifact directory"}:
                held[field] = descriptor
            if field == "frozen artifact directory":
                active = True
            return descriptor, info

        def fail_write(*_args, **_kwargs):
            raise RuntimeError("primary frozen writer failure")

        def tracked_close(descriptor):
            nonlocal injected
            if active:
                close_attempts.append(descriptor)
            original_close(descriptor)
            if (
                active
                and descriptor == held.get("frozen artifact directory")
                and not injected
            ):
                injected = True
                raise OSError("synthetic frozen writer close failure")

        bridge_core._mkdir_private_directory_at = tracked_mkdir
        bridge_core._write_private_file_at = fail_write
        bridge_core.os.close = tracked_close
        caught = None
        try:
            try:
                bridge_core._write_frozen_inputs_fd(staging_fd, expected, ledger)
            except BaseException as exc:  # noqa: BLE001 - inspect cleanup chain
                caught = exc
        finally:
            bridge_core.os.close = original_close
            bridge_core._write_private_file_at = original_write
            bridge_core._mkdir_private_directory_at = original_mkdir
            original_close(staging_fd)
            for descriptor in held.values():
                if descriptor not in close_attempts:
                    try:
                        original_close(descriptor)
                    except OSError:
                        pass
        c.true(injected, "frozen writer close failure was injected")
        c.true(set(held.values()).issubset(set(close_attempts)),
               "frozen writer attempts every held descriptor close")
        c.true(chain_contains(caught, "primary frozen writer failure"),
               "frozen writer cleanup preserves its primary failure")

    cli = _load_cli()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        nested = root / "a" / "b"
        nested.mkdir(parents=True, mode=0o700)
        (root / "a").chmod(0o700)
        chain = cli._open_directory_chain_nofollow(nested)
        original_close = cli.os.close
        close_attempts: list[int] = []
        fail_descriptor = chain.descriptors[-1]
        injected = False

        def tracked_close(descriptor):
            nonlocal injected
            close_attempts.append(descriptor)
            original_close(descriptor)
            if descriptor == fail_descriptor and not injected:
                injected = True
                raise OSError("synthetic directory-chain close failure")

        cli.os.close = tracked_close
        caught = None
        try:
            try:
                cli._close_directory_chain(chain)
            except BaseException as exc:  # noqa: BLE001 - inspect all attempts
                caught = exc
        finally:
            cli.os.close = original_close
            for descriptor in chain.descriptors:
                if descriptor not in close_attempts:
                    try:
                        original_close(descriptor)
                    except OSError:
                        pass
        c.true(injected and isinstance(caught, OSError))
        c.eq(set(close_attempts), set(chain.descriptors),
             "directory-chain cleanup attempts every close after one failure")


def test_multiroot_privacy_cleanup_attempts_every_directory_chain(c: Check):
    cli = _load_cli()
    forbidden = _privacy_scanner_fixture(cli)
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary).resolve()
        roots = (base / "private-a", base / "private-b")
        for root in roots:
            root.mkdir(mode=0o700)

        original_open = cli._open_directory_chain_nofollow
        original_close = cli._close_directory_chain
        opened_chains = []
        close_attempts: list[int] = []
        injected = False

        def tracked_open(path):
            chain = original_open(path)
            opened_chains.append(chain)
            return chain

        def tracked_close(chain):
            nonlocal injected
            close_attempts.append(chain.root_fd)
            original_close(chain)
            if not injected:
                injected = True
                raise OSError("synthetic first privacy-root close failure")

        cli._open_directory_chain_nofollow = tracked_open
        cli._close_directory_chain = tracked_close
        caught: BaseException | None = None
        try:
            try:
                cli._scan_private_trees(roots, forbidden=forbidden)
            except BaseException as exc:  # noqa: BLE001 - inspect cleanup path
                caught = exc
        finally:
            cli._close_directory_chain = original_close
            cli._open_directory_chain_nofollow = original_open
            for chain in opened_chains:
                for descriptor in chain.descriptors:
                    try:
                        cli.os.close(descriptor)
                    except OSError:
                        pass

        c.true(injected and isinstance(caught, OSError),
               "privacy cleanup close failure is observable")
        c.eq(set(close_attempts), {chain.root_fd for chain in opened_chains},
             "privacy cleanup attempts every root chain after one close failure")


def test_transaction_anchors_reject_permission_changes_while_held(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        output = parent / "bridge"
        changed = False

        def chmod_parent_then_authorize():
            nonlocal changed
            if not changed:
                parent.chmod(0o777)
                changed = True
            return ravdess

        observed = None
        try:
            observed = _caught(lambda: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=chmod_parent_then_authorize,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ))
        finally:
            parent.chmod(0o700)
        c.true(changed and isinstance(observed, ValueError),
               "bridge parent chmod during authorization fails closed")
        c.true(not output.exists(),
               "unsafe held bridge parent publishes no canonical generation")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        run_parent = root / "smoke"
        run_parent.mkdir(mode=0o700)
        run = run_parent / "held-mode"
        changed = False

        def chmod_run_parent_then_authorize():
            nonlocal changed
            if not changed:
                run_parent.chmod(0o777)
                changed = True
            return ravdess

        observed = None
        try:
            observed = _caught(lambda: bridge_core.freeze_bridge_stage(
                run,
                bridge,
                mode="smoke",
                ravdess_authorizer=chmod_run_parent_then_authorize,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ))
        finally:
            run_parent.chmod(0o700)
        c.true(changed and isinstance(observed, ValueError),
               "freeze run parent chmod during authorization fails closed")
        c.true(not (run / "inputs").exists(),
               "unsafe held run parent publishes no frozen inputs")

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        key = parent / "key"
        original_fsync = bridge_core._fsync_directory_fd
        changed = False

        def chmod_key_parent_after_fsync(descriptor):
            nonlocal changed
            original_fsync(descriptor)
            if key.exists() and not changed:
                parent.chmod(0o777)
                changed = True

        bridge_core._fsync_directory_fd = chmod_key_parent_after_fsync
        observed = None
        try:
            observed = _caught(lambda: bridge_core.initialize_owner_only_key(key))
        finally:
            bridge_core._fsync_directory_fd = original_fsync
            parent.chmod(0o700)
        c.true(changed and isinstance(observed, ValueError),
               "key parent chmod after publication fsync fails closed")
        c.true(key.is_file(),
               "postpublication key permission fault retains canonical evidence")


def test_concurrent_bridge_and_freeze_transactions_serialize_before_authorization(
    c: Check,
):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64

    def run_pair(operation, lock_name: str):
        original_lock = bridge_core._acquire_destination_lock_at
        barrier = threading.Barrier(2)
        authorizer_calls = 0
        calls_guard = threading.Lock()
        results: list[object] = []

        def synchronized_lock(parent_fd, name, *, create, exclusive, field):
            if name == lock_name and exclusive:
                barrier.wait(timeout=5)
            return original_lock(
                parent_fd, name, create=create, exclusive=exclusive, field=field,
            )

        def ravdess_authorizer():
            nonlocal authorizer_calls
            with calls_guard:
                authorizer_calls += 1
            return ravdess

        def worker():
            try:
                results.append(operation(ravdess_authorizer))
            except BaseException as exc:  # noqa: BLE001 - compare both callers
                results.append(exc)

        bridge_core._acquire_destination_lock_at = synchronized_lock
        try:
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            c.true(all(not thread.is_alive() for thread in threads),
                   "concurrent transaction callers do not deadlock")
        finally:
            bridge_core._acquire_destination_lock_at = original_lock
        c.eq(sum(isinstance(item, dict) for item in results), 1,
             "exactly one concurrent transaction publishes")
        c.eq(sum(isinstance(item, FileExistsError) for item in results), 1,
             "the serialized loser observes the committed destination")
        c.eq(authorizer_calls, 2,
             "only the winning transaction reaches two-pass authorization")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        output = root / "bridge"
        run_pair(
            lambda authorize: bridge_core.build_bridge_bundles(
                output,
                ravdess_authorizer=authorize,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ),
            ".bridge.lock",
        )
        c.true(output.is_dir())
        c.eq(tuple(root.glob(".bridge.staging-*")), (),
             "serialized bridge build leaves no losing staging residue")
        bridge_lock = root / ".bridge.lock"
        c.true(bridge_lock.is_file())
        c.eq(stat.S_IMODE(bridge_lock.stat().st_mode), 0o600)

        run_parent = root / "smoke"
        run = run_parent / "concurrent"
        run_pair(
            lambda authorize: bridge_core.freeze_bridge_stage(
                run,
                output,
                mode="smoke",
                ravdess_authorizer=authorize,
                mayo_authorizer=lambda: mayo,
                producer_sha256=producer,
            ),
            ".inputs.lock",
        )
        c.true((run / "inputs").is_dir())
        c.eq(tuple(run.glob(".inputs.staging-*")), (),
             "serialized freeze leaves no losing staging residue")
        inputs_lock = run / ".inputs.lock"
        c.true(inputs_lock.is_file())
        c.eq(stat.S_IMODE(inputs_lock.stat().st_mode), 0o600)
        c.eq(tuple(run_parent.glob(".*.inputs.lock")), (),
             "first-use serialization adds no sibling lock contract")


def test_frozen_verifier_rejects_sibling_staging_residue(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        run_parent = root / "smoke"
        run_parent.mkdir(mode=0o700)
        run = run_parent / "residue"
        bridge_core.freeze_bridge_stage(
            run,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        foreign = run / ".inputs.staging-foreign"
        foreign.mkdir(mode=0o700)
        c.raises(lambda: bridge_core.verify_frozen_bridge_stage(
            run / "inputs",
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "frozen verifier rejects sibling staging residue")


def test_verifiers_require_existing_exact_persistent_locks_before_authorization(
    c: Check,
):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        bridge = root / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        lock = root / ".bridge.lock"
        scenarios = ("missing", "mode", "nonempty", "hardlink")
        for scenario in scenarios:
            calls = 0
            alias = root / ".bridge.lock.alias"
            if scenario == "missing":
                lock.rename(alias)
            elif scenario == "mode":
                lock.chmod(0o644)
            elif scenario == "nonempty":
                lock.write_bytes(b"unsafe")
            else:
                os.link(lock, alias)

            def authorize():
                nonlocal calls
                calls += 1
                return ravdess

            try:
                c.raises(lambda: bridge_core.verify_bridge_generation(
                    bridge,
                    ravdess_authorizer=authorize,
                    mayo_authorizer=lambda: mayo,
                    producer_sha256=producer,
                ), ValueError, f"bridge verifier rejects {scenario} lock storage")
                c.eq(calls, 0, "unsafe lock fails before live authorization")
            finally:
                if scenario == "missing":
                    alias.rename(lock)
                elif scenario == "mode":
                    lock.chmod(0o600)
                elif scenario == "nonempty":
                    lock.write_bytes(b"")
                    lock.chmod(0o600)
                else:
                    alias.unlink()
            c.true(lock.is_file(), "verifier never creates or replaces the lock")

        run_parent = root / "smoke"
        run_parent.mkdir(mode=0o700)
        run = run_parent / "missing-lock"
        bridge_core.freeze_bridge_stage(
            run,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        inputs_lock = run / ".inputs.lock"
        inputs_lock.unlink()
        calls = 0

        def authorize_frozen():
            nonlocal calls
            calls += 1
            return ravdess

        c.raises(lambda: bridge_core.verify_frozen_bridge_stage(
            run / "inputs",
            bridge,
            mode="smoke",
            ravdess_authorizer=authorize_frozen,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        ), ValueError, "frozen verifier rejects a missing persistent lock")
        c.eq(calls, 0, "missing inputs lock fails before live authorization")
        c.true(not inputs_lock.exists(), "frozen verifier does not recreate storage")


def test_cli_verifier_validates_all_locks_before_inventory_or_authorization(c: Check):
    ravdess, mayo = _synthetic_authorizations()
    producer = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        pretraining = parent / "pretraining"
        pretraining.mkdir(mode=0o700)
        bridge = pretraining / "bridge"
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        run_parent = pretraining / "smoke"
        run_parent.mkdir(mode=0o700)
        run_root = run_parent / "lock-order"
        bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode="smoke",
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        common, _mayo_root, _legacy_root = _cli_common_args(parent)

        def run_missing_lock(lock: Path, run_argument: bool) -> None:
            cli = _load_cli()
            cli.PRETRAINING_ROOT = pretraining
            cli.CANONICAL_MAYO_KEY = pretraining / ".mayo_ssl_hmac.key"
            cli._producer_sha256 = lambda: producer
            inventory_calls = 0
            authorization_calls = 0

            def inventory(_args):
                nonlocal inventory_calls
                inventory_calls += 1
                return SimpleNamespace(member_sha256={"opaque.csv": "7" * 64}), SimpleNamespace()

            def authorize_ravdess():
                nonlocal authorization_calls
                authorization_calls += 1
                return ravdess

            def authorize_mayo():
                nonlocal authorization_calls
                authorization_calls += 1
                return mayo

            cli._live_privacy_inventories = inventory
            cli._authorization_factories = lambda _args: (
                authorize_ravdess, authorize_mayo,
            )
            parked = lock.with_name(lock.name + ".parked")
            lock.rename(parked)
            arguments = [
                "verify-determinism", *common, "--bridge-root", str(bridge),
            ]
            if run_argument:
                arguments.extend(("--run-root", str(run_root)))
            try:
                c.raises(
                    lambda: cli.main(arguments),
                    ValueError,
                    "missing persistent verifier lock fails closed",
                )
            finally:
                parked.rename(lock)
            c.eq(inventory_calls, 0, "lock validation precedes privacy inventory")
            c.eq(authorization_calls, 0, "lock validation precedes data authorization")

        run_missing_lock(pretraining / ".bridge.lock", False)
        run_missing_lock(run_root / ".inputs.lock", True)


if __name__ == "__main__":
    run_all("test_dynamic_landmark_ssl_bridge", dict(globals()))
