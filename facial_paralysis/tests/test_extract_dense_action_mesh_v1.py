from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from scripts.extract_dense_action_mesh_v1 import (
    DENSE_CACHE_SCHEMA,
    build_dense_action_cache,
    extract_normalized_pair,
    load_dense_action_cache_bytes,
    publish_dense_action_cache,
    serialize_dense_action_cache,
)


def _arrays():
    rng = np.random.default_rng(17)
    original = rng.normal(size=(2, 8, 478, 3)).astype(np.float64)
    mirrored = rng.normal(size=(2, 8, 478, 3)).astype(np.float64)
    original_baseline = rng.normal(size=(2, 4, 478, 3)).astype(np.float64)
    mirrored_baseline = rng.normal(size=(2, 4, 478, 3)).astype(np.float64)
    action_valid = np.ones((2, 8), dtype=bool)
    baseline_valid = np.ones((2, 4), dtype=bool)
    action_valid[1, 2] = False
    original[1, 2] = np.nan
    mirrored[1, 2] = np.nan
    return (
        original,
        mirrored,
        original_baseline,
        mirrored_baseline,
        action_valid,
        baseline_valid,
    )


def _cache():
    arrays = _arrays()
    return build_dense_action_cache(
        recording_id="rec_" + "1" * 64,
        group_id="grp_" + "2" * 64,
        source_sha256="3" * 64,
        timing_sha256="4" * 64,
        face_landmarker_sha256="5" * 64,
        action_names=("SMILE", "EYE"),
        source_frame_count=80,
        fps=30.0,
        action_frame_indices=np.stack((np.arange(8), np.arange(20, 28))).astype(np.int64),
        baseline_frame_indices=np.stack((np.arange(8, 12), np.arange(28, 32))).astype(np.int64),
        original_actions=arrays[0],
        mirrored_actions=arrays[1],
        original_baselines=arrays[2],
        mirrored_baselines=arrays[3],
        action_valid=arrays[4],
        baseline_valid=arrays[5],
    )


def test_cache_round_trip_is_exact_deterministic_and_immutable(c):
    cache = _cache()
    first = serialize_dense_action_cache(cache)
    second = serialize_dense_action_cache(cache)
    c.eq(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())
    loaded = load_dense_action_cache_bytes(first)
    c.eq(loaded.schema_version, DENSE_CACHE_SCHEMA)
    c.eq(loaded.action_names, ("SMILE", "EYE"))
    c.true(np.array_equal(loaded.action_frame_indices, cache.action_frame_indices))
    c.true(np.isnan(loaded.original_actions[1, 2]).all())
    c.true(not loaded.original_actions.flags.writeable)


def test_cache_rejects_source_timing_schema_and_mask_tampering(c):
    kwargs = dict(_cache().__dict__)
    for name, value in (
        ("source_sha256", "bad"),
        ("timing_sha256", "6" * 63),
        ("action_names", ("SMILE", "SMILE")),
    ):
        changed = dict(kwargs)
        changed[name] = value
        c.raises(lambda fields=changed: build_dense_action_cache(**fields), ValueError)
    changed = dict(kwargs)
    bad_mask = kwargs["action_valid"].copy()
    bad_mask[0, :3] = False
    changed["action_valid"] = bad_mask
    c.raises(lambda: build_dense_action_cache(**changed), ValueError)


def test_masked_misses_stay_nan_and_valid_rows_cannot_be_nonfinite(c):
    kwargs = dict(_cache().__dict__)
    corrupted = kwargs["original_actions"].copy()
    corrupted[0, 0] = np.nan
    kwargs["original_actions"] = corrupted
    c.raises(lambda: build_dense_action_cache(**kwargs), ValueError)


def test_actual_original_and_flipped_frames_are_detected_independently(c):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :5, 0] = 255
    calls = []

    def detector(rgb):
        calls.append(rgb.copy())
        mesh = np.zeros((478, 3), dtype=np.float64)
        mesh[:, 0] = np.linspace(0.2, 0.8, 478)
        mesh[:, 1] = np.linspace(0.3, 0.7, 478)
        mesh[33, :2] = (0.3, 0.4)
        mesh[263, :2] = (0.7, 0.4)
        mesh[61, 1] = float(rgb[0, 0, 2] > 0)
        return mesh

    original, mirrored = extract_normalized_pair(frame, detector)
    c.eq(len(calls), 2)
    c.true(np.array_equal(calls[1], calls[0][:, ::-1]))
    c.true(not np.array_equal(original, mirrored))


def test_private_publication_is_mode_0600_and_no_overwrite(c):
    payload = serialize_dense_action_cache(_cache())
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "cache.npz"
        digest = publish_dense_action_cache(output, payload)
        c.eq(digest, hashlib.sha256(payload).hexdigest())
        c.eq(os.stat(output).st_mode & 0o777, 0o600)
        c.eq(output.read_bytes(), payload)
        c.raises(lambda: publish_dense_action_cache(output, payload), FileExistsError)


if __name__ == "__main__":
    run_all("test_extract_dense_action_mesh_v1", dict(globals()))
