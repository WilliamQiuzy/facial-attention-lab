"""Feature-layout tests for blendshape/landmark fusion without loading MediaPipe."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.action_bundle import (  # noqa: E402
    MediaPipeFeatureExtractor,
    _assert_existing_cache_schema,
    _bundle_npz_payload,
)
from src.preprocessing.clinical_landmarks import CLINICAL_LANDMARK_NAMES  # noqa: E402
from src.datasets.patient_multistream import (  # noqa: E402
    MP_FEATURE_NAMES_BY_SCHEMA,
    MP_SIDE_CONVENTION_BY_SCHEMA,
)
from _testlib import Check, run_all  # noqa: E402
from test_clinical_landmarks import _face  # noqa: E402


def _extractor(mode: str):
    ext = object.__new__(MediaPipeFeatureExtractor)
    ext.landmark_features = mode
    ext._bs_names = ["_neutral", "mouthSmileLeft", "mouthSmileRight"]
    ext._pairs = [(1, 2)]
    return ext


def _registered_extractor(mode: str):
    ext = object.__new__(MediaPipeFeatureExtractor)
    ext.landmark_features = mode
    ext.capture_mirrored = None
    ext._bs_names = None
    ext._pairs = None
    ext._init_layout(list(MP_FEATURE_NAMES_BY_SCHEMA["mediapipe_bs_lr_v1"][:52]))
    return ext


def _landmark_objects():
    return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in _face()]


def test_none_mode_preserves_blendshape_layout(c: Check):
    ext = _extractor("none")
    v = ext._assemble_features(np.array([0.0, 0.7, 0.2], np.float32), None, 640, 480)
    c.eq(ext.feat_dim, 4, "three blendshapes plus one mirrored delta")
    c.eq(ext.feature_names, ["_neutral", "mouthSmileLeft", "mouthSmileRight",
                             "delta_left_minus_right_mouthSmile"])
    c.true(bool(np.allclose(v, [0.0, 0.7, 0.2, 0.5])), "signed left-right delta")
    c.eq(ext.feature_schema, "mediapipe_bs_lr_v1", "stable legacy schema")


def test_clinical23_mode_appends_landmarks(c: Check):
    ext = _extractor("clinical23")
    v = ext._assemble_features(
        np.array([0.0, 0.7, 0.2], np.float32), _landmark_objects(), 1000, 1000)
    c.eq(ext.feat_dim, 4 + 23, "fused dimension")
    c.eq(v.shape, (27,), "fused vector")
    c.eq(ext.feature_names[-23:], list(CLINICAL_LANDMARK_NAMES), "clinical names appended")
    c.eq(ext.feature_schema, "mediapipe_bs_lr_v1+clinical23_v2", "fused schema")


def test_landmarks_required_only_for_landmark_modes(c: Check):
    base = _extractor("none")
    c.true(base._assemble_features(np.zeros(3, np.float32), None, 10, 10) is not None,
           "blendshape-only accepts absent landmarks")
    clinical = _extractor("clinical23")
    c.raises(lambda: clinical._assemble_features(np.zeros(3, np.float32), None, 10, 10),
             ValueError, "clinical mode requires landmarks")


def test_mode_validation(c: Check):
    c.raises(lambda: MediaPipeFeatureExtractor._validate_landmark_features("dense999"),
             ValueError, "unknown mode rejected")


def test_all_missed_frames_keep_known_feature_width(c: Check):
    ext = _extractor("clinical23")
    ext._frame_features = lambda _frame: None
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    seq, mask = ext.extract_sequence(frames)
    c.eq(seq.shape, (3, 27), "known layout produces correctly-sized zero rows")
    c.true(bool((seq == 0).all()), "missed rows are zero")
    c.true(bool((~mask).all()), "all missed rows stay masked")


def test_all_missed_frames_without_layout_mark_mp_absent(c: Check):
    ext = object.__new__(MediaPipeFeatureExtractor)
    ext.landmark_features = "clinical23"
    ext._bs_names = None
    ext._pairs = None
    ext._frame_features = lambda _frame: None
    seq, mask = ext.extract_sequence([np.zeros((8, 8, 3), np.uint8)])
    c.eq(seq.shape, (1, 0), "unknown layout is explicit MP-stream absence")
    c.true(bool((~mask).all()), "absent stream is masked")


def test_npz_payload_rejects_declared_width_mismatch(c: Check):
    ext = _extractor("clinical23")
    bad = {
        "marlin": np.zeros((1, 768), np.float32),
        "mp_seq": np.zeros((2, 1), np.float32),
        "mp_mask": np.zeros(2, bool),
    }
    c.raises(lambda: _bundle_npz_payload(bad, ext), ValueError,
             "writer rejects a (T,1) stream that declares 27 features")


def test_npz_payload_omits_unknown_mp_stream(c: Check):
    ext = object.__new__(MediaPipeFeatureExtractor)
    ext.landmark_features = "clinical23"
    ext._bs_names = None
    ext._pairs = None
    bundle = {
        "marlin": np.zeros((1, 768), np.float32),
        "mp_seq": np.zeros((2, 0), np.float32),
        "mp_mask": np.zeros(2, bool),
    }
    payload = _bundle_npz_payload(bundle, ext)
    c.eq(sorted(payload), ["marlin"], "MARLIN-only cache contains no fake MP metadata")


def test_first_blendshape_layout_must_match_registered_schema(c: Check):
    ext = object.__new__(MediaPipeFeatureExtractor)
    ext.landmark_features = "none"
    ext._bs_names = None
    ext._pairs = None
    c.raises(
        lambda: ext._ensure_layout(["not_a_mediapipe_category"]),
        RuntimeError,
        "the first detector frame cannot mint an invalid named schema",
    )


def test_npz_payload_rejects_nonfinite_valid_rows(c: Check):
    ext = _registered_extractor("clinical23")
    seq = np.zeros((2, ext.feat_dim), np.float32)
    seq[0, 0] = np.nan
    c.raises(
        lambda: _bundle_npz_payload({
            "marlin": np.zeros((1, 768), np.float32),
            "mp_seq": seq,
            "mp_mask": np.asarray((True, False)),
        }, ext),
        ValueError,
        "a detector-valid frame must be entirely finite",
    )


def test_npz_payload_canonicalizes_nonfinite_masked_rows(c: Check):
    ext = _registered_extractor("clinical23")
    seq = np.zeros((2, ext.feat_dim), np.float32)
    seq[1] = np.nan
    payload = _bundle_npz_payload({
        "marlin": np.zeros((1, 768), np.float32),
        "mp_seq": seq,
        "mp_mask": np.asarray((True, False)),
    }, ext)
    c.true(np.isfinite(payload["mp_seq"]).all(), "serialized action cache is finite")
    c.true((payload["mp_seq"][1] == 0).all(), "masked frame is canonical zero padding")


def test_npz_payload_rejects_nonfinite_marlin(c: Check):
    ext = _registered_extractor("none")
    marlin = np.zeros((1, 768), np.float32)
    marlin[0, 0] = np.inf
    c.raises(
        lambda: _bundle_npz_payload({
            "marlin": marlin,
            "mp_seq": np.zeros((1, ext.feat_dim), np.float32),
            "mp_mask": np.ones(1, bool),
        }, ext),
        ValueError,
        "nonfinite MARLIN embeddings cannot enter a cache",
    )


def test_existing_cache_must_match_requested_schema(c: Check):
    with tempfile.TemporaryDirectory() as tmp:
        marlin_only = Path(tmp) / "marlin_only.npz"
        legacy_mp = Path(tmp) / "legacy_mp.npz"
        wrong = Path(tmp) / "wrong_schema.npz"
        right = Path(tmp) / "right_schema.npz"
        np.savez(marlin_only, marlin=np.zeros((1, 2), np.float32))
        np.savez(legacy_mp, marlin=np.zeros((1, 2), np.float32),
                 mp_seq=np.zeros((2, 72), np.float32), mp_mask=np.ones(2, bool))
        np.savez(wrong, mp_feature_schema=np.asarray("mediapipe_bs_lr_v1"))
        schema = "mediapipe_bs_lr_v1+clinical23_v2"
        ext = object.__new__(MediaPipeFeatureExtractor)
        ext.landmark_features = "clinical23"
        ext._bs_names = None
        ext._pairs = None
        ext._init_layout(list(MP_FEATURE_NAMES_BY_SCHEMA[schema][:52]))
        payload = _bundle_npz_payload({
            "marlin": np.zeros((1, 2), np.float32),
            "mp_seq": np.zeros((2, ext.feat_dim), np.float32),
            "mp_mask": np.ones(2, bool),
        }, ext)
        np.savez(right, **payload)
        _assert_existing_cache_schema(marlin_only, schema)
        c.raises(lambda: _assert_existing_cache_schema(
            legacy_mp, schema), RuntimeError,
            "metadata-free MP cache requires overwrite")
        c.raises(lambda: _assert_existing_cache_schema(
            wrong, schema), RuntimeError,
            "mixed schema requires overwrite")
        _assert_existing_cache_schema(right, schema)


def test_writer_and_loader_share_exact_clinical_schema(c: Check):
    schema = "mediapipe_bs_lr_v1+clinical23_v2"
    expected = MP_FEATURE_NAMES_BY_SCHEMA[schema]
    ext = object.__new__(MediaPipeFeatureExtractor)
    ext.landmark_features = "clinical23"
    ext._bs_names = None
    ext._pairs = None
    ext._init_layout(list(expected[:52]))
    c.eq(ext.feature_schema, schema, "writer emits the registered schema id")
    c.eq(tuple(ext.feature_names), expected,
         "writer column order exactly matches the loader registry")
    c.eq(ext.side_convention, MP_SIDE_CONVENTION_BY_SCHEMA[schema],
         "writer side provenance matches the loader registry")


def test_every_frame_revalidates_blendshape_column_order(c: Check):
    ext = _extractor("none")
    ext._ensure_layout(["_neutral", "mouthSmileLeft", "mouthSmileRight"])
    c.raises(lambda: ext._ensure_layout(
        ["_neutral", "mouthSmileRight", "mouthSmileLeft"]), RuntimeError,
        "category reordering cannot silently corrupt feature columns")


if __name__ == "__main__":
    run_all("test_action_bundle_landmarks", dict(globals()))
