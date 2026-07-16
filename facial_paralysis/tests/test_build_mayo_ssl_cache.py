"""Contract tests for the transactional, deidentified Mayo SSL cache builder."""
from __future__ import annotations

import csv
import base64
import hashlib
import io
import inspect
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)

SCRIPT = ROOT / "scripts" / "build_mayo_ssl_cache.py"
SPEC = importlib.util.spec_from_file_location("build_mayo_ssl_cache", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Mayo SSL builder module")
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


EXISTING_FRAME_COUNTS = (
    5536, 7376, 7140, 7038, 4579, 5604, 6962,
    5313, 4021, 5029, 6580, 7175, 8529,
)
PENDING_FRAME_COUNTS = (
    6021, 6779, 5210, 5576, 7040, 4758, 4444, 5524, 5384,
    6780, 4952, 4861, 6769, 6900, 6583, 5471, 4502, 6738,
    5641, 5312, 7442, 7094, 6858, 6877, 6730, 6632, 6801,
    6673, 7339, 6778, 6916, 7731, 7114, 7338, 7553,
)
ARKIT_ROW_COUNTS = (8435, 7433, 8356, 7367, 6892, 6042, 6218, 7311)
ARKIT_GAP_COUNTS = (3, 3, 3, 4, 3, 2, 3, 3)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prepend_duplicate_json_field(
    payload: bytes,
    field: str,
    hidden_value: object,
) -> bytes:
    text = payload.decode("utf-8")
    marker = f'"{field}":'
    if marker not in text:
        raise AssertionError(f"fixture has no JSON field {field!r}")
    duplicate = f'{marker}{json.dumps(hidden_value)},{marker}'
    return text.replace(marker, duplicate, 1).encode("utf-8")


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _write_complete_export(root: Path, session_name: str) -> None:
    target = root / session_name
    target.mkdir(parents=True)
    (target / "done.json").write_text("{}")
    (target / "landmarks.csv").write_text("frame,point_idx,x,y,z\n")
    (target / "blendshapes_wide.csv").write_text("frame,timestamp_ms\n")
    np.save(target / "transform_matrices.npy", np.empty((0, 4, 4), np.float32))


def _inventory_fixture(root: Path):
    data = root / "PHI_raw_mayo"
    exports = root / "PHI_existing_exports"
    data.mkdir()
    exports.mkdir()
    video_meta: dict[Path, object] = {}
    arkit_rows: dict[Path, int] = {}
    arkit_gaps: dict[Path, int] = {}

    representative_payload = b"exact-duplicate-video"
    representative_frames = EXISTING_FRAME_COUNTS[3]
    for index, frame_count in enumerate(EXISTING_FRAME_COUNTS):
        name = f"PHI_existing_{index:02d}"
        session = data / name
        session.mkdir()
        payload = representative_payload if index == 3 else f"existing-{index}".encode()
        video = session / f"private_{index}.mov"
        video.write_bytes(payload)
        video_meta[video] = builder.VideoMetadata(
            frame_count=frame_count, fps=60.0, width=720, height=1280
        )
        _write_complete_export(exports, name)
        if index < 2:
            (session / f"private_{index}.mp4").write_bytes(b"proxy-not-a-recording")

    for index, frame_count in enumerate(PENDING_FRAME_COUNTS):
        name = f"PHI_pending_{index:02d}"
        session = data / name
        session.mkdir()
        video = session / f"private_pending_{index}.mov"
        video.write_bytes(f"pending-{index}".encode())
        video_meta[video] = builder.VideoMetadata(
            frame_count=frame_count, fps=60.0, width=720, height=1280
        )
        if index == 0:
            (session / "private_pending_0.mp4").write_bytes(b"proxy")

    duplicate_name = "ZZ_PHI_duplicate_copy"
    duplicate = data / duplicate_name
    duplicate.mkdir()
    duplicate_video = duplicate / "duplicate.mov"
    duplicate_video.write_bytes(representative_payload)
    video_meta[duplicate_video] = builder.VideoMetadata(
        frame_count=representative_frames, fps=60.0, width=720, height=1280
    )
    _write_complete_export(exports, duplicate_name)

    short_name = "ZZ_PHI_qc_short"
    short = data / short_name
    short.mkdir()
    short_video = short / "short.mov"
    short_video.write_bytes(b"short-qc")
    video_meta[short_video] = builder.VideoMetadata(
        frame_count=68, fps=60.0, width=720, height=1280
    )
    _write_complete_export(exports, short_name)

    row_index = 0
    for session_index in range(7):
        session = data / f"PHI_arkit_only_{session_index:02d}"
        session.mkdir()
        trajectory_count = 2 if session_index == 5 else 1
        for trajectory_index in range(trajectory_count):
            csv_path = session / f"private_trajectory_{trajectory_index}_iPhone.csv"
            csv_path.write_text(
                f"Timecode,BlendshapeCount\nopaque-{row_index},61\n"
            )
            arkit_rows[csv_path] = ARKIT_ROW_COUNTS[row_index]
            arkit_gaps[csv_path] = ARKIT_GAP_COUNTS[row_index]
            row_index += 1

    for index in range(8):
        session = data / f"PHI_metadata_only_{index:02d}"
        session.mkdir()
        (session / "frame_log.csv").write_text("kind,index\nV,0\n")
        (session / "video_metadata.json").write_text('{"FrameRate":60}')

    def probe(path: Path):
        return video_meta[Path(path)]

    def inspect_arkit(path: Path):
        return builder.ARKitInspection(
            row_count=arkit_rows[Path(path)],
            feature_names=builder.ARKIT_BLENDSHAPE_NAMES,
            missing_source_frames=arkit_gaps[Path(path)],
        )

    return data, exports, probe, inspect_arkit


def _inventory(root: Path):
    data, exports, probe, inspect_arkit = _inventory_fixture(root)
    inventory = builder.inventory_mayo_sources(
        data,
        exports,
        probe_video=probe,
        inspect_arkit=inspect_arkit,
        enforce_frozen=True,
    )
    return data, exports, inventory, probe, inspect_arkit


def test_frozen_inventory_classifies_video_exports_and_arkit_pool(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _data, _exports, inventory, _probe, _inspect = _inventory(Path(td))
        c.eq(inventory.counts, builder.FROZEN_INVENTORY,
             "observed inventory exactly matches the frozen 65-session contract")
        c.eq(len(inventory.video_instances), 50)
        c.eq(len(inventory.long_unique_videos), 48)
        c.eq(len(inventory.existing_export_videos), 13)
        c.eq(sum(item.metadata.frame_count for item in inventory.pending_videos), 221_121)
        c.eq(len(inventory.duplicate_videos), 1)
        c.eq(len(inventory.short_videos), 1)
        c.true(abs(inventory.short_videos[0].metadata.duration_seconds - 1.1333333333) < 1e-8)
        c.eq(len(inventory.arkit_sessions), 7)
        c.eq(len(inventory.arkit_trajectories), 8)
        c.eq(sum(item.row_count for item in inventory.arkit_trajectories), 58_054)
        c.eq(inventory.counts["arkit_timecode_gaps"], 24,
             "the frozen auxiliary audit preserves all 24 missing source frames")
        c.eq(len(inventory.metadata_only_sessions), 8)


def test_inventory_drift_is_reported_not_silently_reclassified(c: Check):
    with tempfile.TemporaryDirectory() as td:
        data, exports, _inventory_ok, probe, inspect_arkit = _inventory(Path(td))
        unexpected_clip = data / "PHI_pending_00" / "unrelated_second_clip.mp4"
        unexpected_clip.write_bytes(b"not-a-proxy")
        c.raises(lambda: builder.inventory_mayo_sources(
            data, exports, probe_video=probe, inspect_arkit=inspect_arkit,
            enforce_frozen=True,
        ), ValueError, "an unrelated second clip cannot hide behind canonical MOV preference")
        unexpected_clip.unlink()
        (data / "PHI_unexpected_session").mkdir()
        c.raises(lambda: builder.inventory_mayo_sources(
            data, exports, probe_video=probe, inspect_arkit=inspect_arkit,
            enforce_frozen=True,
        ), builder.InventoryDriftError, "a 66th session trips the frozen-inventory gate")


def _sequence_with_gap() -> object:
    source = np.asarray([0, 1, 2, 5, 6, 7], dtype=np.int64)
    features = np.repeat(source[:, None], 95, axis=1).astype(np.float32)
    mask = np.asarray([True, True, True, True, True, False], dtype=bool)
    features[~mask] = 0.0
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], len(source), axis=0)
    transform_mask = mask.copy()
    transforms[~transform_mask] = 0.0
    return builder.MayoMediaSequence(
        features=features,
        valid_mask=mask,
        timestamps=source.astype(np.float64) / 60.0,
        source_frame_indices=source,
        facial_transforms=transforms,
        facial_transform_mask=transform_mask,
        transform_source="same_detection_mediapipe_video_mode",
    )


def test_deterministic_30hz_view_selects_without_interpolating_gaps(c: Check):
    view = builder.downsample_60hz_to_30hz(_sequence_with_gap())
    c.eq(view.source_frame_indices.tolist(), [0, 2, 6],
         "30-Hz view selects an exact source-frame phase")
    c.eq(view.timestamps.tolist(), [0.0, 2.0 / 60.0, 6.0 / 60.0])
    c.eq(view.features[:, 0].tolist(), [0.0, 2.0, 6.0])
    c.eq(view.contiguous_from_previous.tolist(), [False, True, False],
         "the source gap is explicit and never bridged")

    base = _sequence_with_gap()
    transforms = base.facial_transforms.copy()
    transform_mask = base.facial_transform_mask.copy()
    transforms[-1] = np.eye(4, dtype=np.float32)
    transform_mask[-1] = True
    inconsistent = builder.MayoMediaSequence(
        features=base.features,
        valid_mask=base.valid_mask,
        timestamps=base.timestamps,
        source_frame_indices=base.source_frame_indices,
        facial_transforms=transforms,
        facial_transform_mask=transform_mask,
        transform_source=base.transform_source,
    )
    c.raises(lambda: builder.downsample_to_30hz(inconsistent), ValueError,
             "a transform cannot be declared valid when the same detection is invalid")


def test_hmac_manifests_hide_names_and_mark_every_video_exposed(c: Check):
    with tempfile.TemporaryDirectory() as td:
        data, exports, inventory, _probe, _inspect = _inventory(Path(td))
        salt = b"local-secret-salt-for-mayo-ssl-0123456789"
        collection, exposure = builder.build_public_manifests(inventory, salt)
        builder.validate_public_manifest(collection)
        builder.validate_public_manifest(exposure)
        c.raises(lambda: builder.validate_public_manifest(
            {"opaque": "20260305_FACES018"}), ValueError,
                 "a bare Mayo session identifier is a raw-name leak")
        c.eq(len(collection["mediapipe_records"]), 48)
        c.eq(len(collection["arkit_records"]), 8)
        c.eq(len(exposure["videos"]), 50)
        c.true(all(row["development_only"] and row["ssl_exposed"]
                   and not row["independent_evaluation_eligible"]
                   for row in exposure["videos"]))
        serialized = json.dumps((collection, exposure), sort_keys=True)
        raw_source_hashes = {
            asset.source_sha256 for asset in inventory.video_instances
        } | {
            asset.source_sha256 for asset in inventory.arkit_trajectories
        }
        c.true(not any(digest in serialized for digest in raw_source_hashes),
               "public collection and exposure ledgers never serialize raw source hashes")
        c.true("PHI_" not in serialized and str(data) not in serialized
               and str(exports) not in serialized,
               "no session name or filesystem location reaches either manifest")
        c.true(salt.decode() not in serialized, "the local HMAC salt is never serialized")
        duplicate_groups = [row["group_id"] for row in exposure["videos"]
                            if row["status"] == "exact_duplicate_excluded"]
        c.eq(len(duplicate_groups), 1)
        c.true(duplicate_groups[0] in {
            row["group_id"] for row in exposure["videos"]
            if row["status"] != "exact_duplicate_excluded"
        }, "only exact duplicate provenance creates a shared group")
        other_collection, other_exposure = builder.build_public_manifests(
            inventory, b"x" * 40
        )
        c.true(collection["mediapipe_records"][0]["recording_id"]
               != other_collection["mediapipe_records"][0]["recording_id"],
               "opaque identifiers are keyed by the local salt")
        first_integrity_ids = {
            row["source_integrity_id"]
            for row in (*collection["mediapipe_records"], *collection["arkit_records"])
        }
        first_integrity_ids.update(
            row["source_integrity_id"] for row in exposure["videos"]
        )
        other_integrity_ids = {
            row["source_integrity_id"]
            for row in (
                *other_collection["mediapipe_records"],
                *other_collection["arkit_records"],
                *other_exposure["videos"],
            )
        }
        c.true(first_integrity_ids.isdisjoint(other_integrity_ids),
               "per-record integrity bindings are scoped to the local salt")
        simulated_cache_hashes = [
            hashlib.sha256(f"cache-{index}".encode()).hexdigest()
            for index in range(4)
        ]
        first_cache_ids = {
            builder.hmac_identifier(
                "cache", salt, "mayo-mediapipe-cache-integrity", digest
            ) for digest in simulated_cache_hashes
        }
        other_cache_ids = {
            builder.hmac_identifier(
                "cache", b"x" * 40, "mayo-mediapipe-cache-integrity", digest
            ) for digest in simulated_cache_hashes
        }
        c.true(first_cache_ids.isdisjoint(other_cache_ids),
               "cache integrity bindings are also scoped to the local salt")


def test_public_rows_omit_exact_temporal_quasi_identifiers(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _data, _exports, inventory, _probe, _inspect = _inventory(Path(td))
        collection, exposure = builder.build_public_manifests(
            inventory, b"quasi-identifier-salt-012345678901"
        )
        public_text = json.dumps((collection, exposure), sort_keys=True)
        for forbidden in (
            "source_frame_count", "\"fps\"", "rows_60hz",
            "missing_source_frames",
        ):
            c.true(forbidden not in public_text,
                   f"per-record exact temporal metadata {forbidden} stays private")

        pending_rows = [
            row for row in collection["mediapipe_records"]
            if row["legacy_export_audit_status"] == "no_complete_legacy_export"
        ]
        c.true(len(pending_rows) >= 2)
        opaque = {
            "recording_id", "group_id", "source_integrity_id", "source_fingerprint"
        }
        c.eq(
            {key: value for key, value in pending_rows[0].items() if key not in opaque},
            {key: value for key, value in pending_rows[1].items() if key not in opaque},
            "two videos with different exact lengths/FPS are not enumerable publicly",
        )
        arkit_rows = collection["arkit_records"]
        c.true(len(arkit_rows) >= 2)
        c.eq(
            {key: value for key, value in arkit_rows[0].items() if key not in opaque},
            {key: value for key, value in arkit_rows[1].items() if key not in opaque},
            "ARKit row counts and Timecode gaps do not fingerprint public rows",
        )


def _synthetic_face_landmarks():
    points = np.full((478, 3), (0.5, 0.5, 0.0), dtype=np.float32)
    coordinates = {
        33: (0.30, 0.40), 133: (0.40, 0.40),
        159: (0.35, 0.38), 158: (0.37, 0.38), 160: (0.33, 0.38),
        145: (0.35, 0.42), 144: (0.33, 0.42), 153: (0.37, 0.42),
        263: (0.70, 0.40), 362: (0.60, 0.40),
        386: (0.65, 0.38), 385: (0.63, 0.38), 387: (0.67, 0.38),
        374: (0.65, 0.42), 380: (0.67, 0.42), 373: (0.63, 0.42),
        61: (0.40, 0.70), 291: (0.60, 0.70),
        13: (0.50, 0.68), 14: (0.50, 0.72),
    }
    for index, xy in coordinates.items():
        points[index, :2] = xy
    for index, x in zip((70, 63, 105, 66, 107), np.linspace(0.30, 0.40, 5)):
        points[index, :2] = (x, 0.30)
    for index, x in zip((300, 293, 334, 296, 336), np.linspace(0.70, 0.60, 5)):
        points[index, :2] = (x, 0.30)
    for offset, index in enumerate((168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 17, 152, 10)):
        points[index, :2] = (0.5, 0.25 + 0.04 * offset)
    return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in points]


def test_video_mode_extractor_uses_one_detection_for_95d_and_transform(c: Check):
    class FakeMP:
        class ImageFormat:
            SRGB = "srgb"

        class Image:
            def __init__(self, *, image_format, data):
                self.image_format = image_format
                self.data = data

    class VideoOnlyLandmarker:
        def __init__(self):
            self.video_calls = []
            self.closed = 0

        def detect_for_video(self, image, timestamp_ms):
            self.video_calls.append((image, timestamp_ms))
            categories = [
                SimpleNamespace(category_name=name, score=0.01 * (index + 1))
                for index, name in enumerate(DYNAMIC_FEATURE_NAMES[:52])
            ]
            return SimpleNamespace(
                face_blendshapes=[categories],
                face_landmarks=[_synthetic_face_landmarks()],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float32)],
            )

        def close(self):
            self.closed += 1

    landmarker = VideoOnlyLandmarker()
    with tempfile.TemporaryDirectory() as td:
        model = Path(td) / "face_landmarker.task"
        model.write_bytes(b"model")
        extractor = builder.MayoVideoClinical23Extractor(
            model_path=model,
            runtime_factory=lambda _model: (FakeMP, landmarker),
        )
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        features, nuisance, transform = extractor.extract_video_frame(frame, 17)
        c.eq(features.shape, (95,), "same VIDEO detection yields the registered 95-d vector")
        c.eq(tuple(extractor.feature_names), tuple(DYNAMIC_FEATURE_NAMES))
        c.eq(extractor.feature_schema, DYNAMIC_FEATURE_SCHEMA)
        c.eq(transform.shape, (4, 4))
        c.true(bool(np.isfinite(features).all()) and bool(np.isfinite(transform).all()))
        c.true(nuisance is not None, "same landmarks provide nuisance metadata")
        c.eq([call[1] for call in landmarker.video_calls], [17],
             "producer calls detect_for_video, never IMAGE detect")
        extractor.close()
        extractor.close()
        c.eq(landmarker.closed, 1, "VIDEO landmarker closes exactly once")


def test_source_fps_timeline_and_exact_30hz_rows_preserve_gaps(c: Check):
    class FakeCapture:
        def __init__(self):
            self.frames = [np.zeros((4, 5, 3), np.uint8) for _ in range(7)]
            self.index = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 59.95,
                cv2.CAP_PROP_FRAME_COUNT: 7.0,
                cv2.CAP_PROP_FRAME_WIDTH: 5.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 4.0,
            }.get(prop, 0.0)

        def read(self):
            if self.index == len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            self.released = True

    class FakeExtractor:
        feature_schema = DYNAMIC_FEATURE_SCHEMA
        feature_names = list(DYNAMIC_FEATURE_NAMES)
        producer_protocol = "mediapipe_face_landmarker_running_mode_video_v1"

        def __init__(self):
            self.timestamps_ms = []

        def extract_video_frame(self, _frame, timestamp_ms):
            self.timestamps_ms.append(timestamp_ms)
            value = len(self.timestamps_ms)
            return (np.full(95, value, np.float32),
                    {"face_scale": 0.2, "eye_line_roll_degrees": 3.0},
                    np.eye(4, dtype=np.float32))

    capture = FakeCapture()
    extractor = FakeExtractor()
    sequence = builder.extract_video_sequence(
        Path("not-opened-by-the-fake"), extractor,
        capture_factory=lambda _path: capture,
    )
    c.eq(sequence.features.shape, (7, 95))
    c.true(bool(sequence.valid_mask.all()))
    c.eq(sequence.source_fps, 59.95)
    c.true(bool(np.allclose(sequence.timestamps,
                            np.arange(7, dtype=np.float64) / 59.95,
                            rtol=0.0, atol=1e-12)),
           "timestamps retain the audited source FPS rather than pretending 60 Hz")
    c.eq(extractor.timestamps_ms, [0, 17, 33, 50, 67, 83, 100])
    c.true(capture.released, "video capture is released explicitly")
    c.eq(sequence.transform_source, "same_detection_mediapipe_video_mode")
    c.true(bool(sequence.facial_transform_mask.all()))

    gap_indices = np.asarray([0, 1, 2, 5, 6, 7], dtype=np.int64)
    gap_features = np.repeat(gap_indices[:, None], 95, axis=1).astype(np.float32)
    gap_sequence = builder.MayoMediaSequence(
        features=gap_features,
        valid_mask=np.ones(len(gap_indices), dtype=bool),
        timestamps=gap_indices.astype(np.float64) / 59.95,
        source_frame_indices=gap_indices,
        facial_transforms=np.repeat(np.eye(4, dtype=np.float32)[None],
                                    len(gap_indices), axis=0),
        facial_transform_mask=np.ones(len(gap_indices), dtype=bool),
        transform_source="same_detection_mediapipe_video_mode",
        source_fps=59.95,
        timestamp_source="source_frame_index_divided_by_audited_fps",
    )
    view = builder.downsample_to_30hz(gap_sequence)
    c.eq(view.source_frame_indices.tolist(), [0, 2, 6],
         "30-Hz targets select exact source indices and never nearest-fill index 4")
    c.eq(view.target_frame_indices.tolist(), [0, 1, 3])
    c.eq(view.contiguous_from_previous.tolist(), [False, True, False])


def test_homogeneous_materializer_reextracts_assets_with_legacy_exports(c: Check):
    class OneFrameCapture:
        def __init__(self, label):
            self.label = label
            self.read_count = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 59.95,
                cv2.CAP_PROP_FRAME_COUNT: 1.0,
                cv2.CAP_PROP_FRAME_WIDTH: 5.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 4.0,
            }.get(prop, 0.0)

        def read(self):
            if self.read_count:
                return False, None
            self.read_count += 1
            return True, np.zeros((4, 5, 3), np.uint8)

        def release(self):
            self.released = True

    class Extractor:
        feature_schema = DYNAMIC_FEATURE_SCHEMA
        feature_names = list(DYNAMIC_FEATURE_NAMES)

        def __init__(self):
            self.calls = 0
            self.timestamps = []
            self.closed = 0

        def extract_video_frame(self, _frame, timestamp_ms):
            self.calls += 1
            self.timestamps.append(timestamp_ms)
            return (np.full(95, self.calls, np.float32), None,
                    np.eye(4, dtype=np.float32))

        def close(self):
            self.closed += 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        export = root / "legacy_export"
        export.mkdir()
        first = root / "first.mov"
        second = root / "second.mov"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        model = root / "model.task"
        model.write_bytes(b"model")
        assets = (
            builder.VideoAsset(root, first, builder.VideoMetadata(1, 59.95, 5, 4),
                               _sha(b"first"), export),
            builder.VideoAsset(root, second, builder.VideoMetadata(1, 59.95, 5, 4),
                               _sha(b"second"), None),
        )
        captures = {}

        def capture_factory(path):
            captures[path] = OneFrameCapture(path)
            return captures[path]

        extractors = []

        def extractor_factory(**_kwargs):
            item = Extractor()
            extractors.append(item)
            return item

        materialized = list(builder.extract_homogeneous_video_sequences(
            assets, extractor_factory, model_path=model,
            capture_factory=capture_factory,
        ))
        c.eq([asset.path for asset, _sequence in materialized], [first, second])
        c.eq(len(extractors), 2,
             "each VIDEO stream gets fresh tracking state under one producer protocol")
        c.eq([item.timestamps for item in extractors], [[0], [0]],
             "per-video MediaPipe timestamps restart without cross-recording state")
        c.eq([item.closed for item in extractors], [1, 1])
        c.eq(set(captures), {str(first), str(second)})
        c.true(all(capture.released for capture in captures.values()))

        def drifted_capture_factory(path):
            capture = OneFrameCapture(path)
            original_get = capture.get

            def get(prop):
                return 60.0 if prop == cv2.CAP_PROP_FPS else original_get(prop)

            capture.get = get
            return capture

        c.raises(lambda: list(builder.extract_homogeneous_video_sequences(
            assets[:1], extractor_factory, model_path=model,
            capture_factory=drifted_capture_factory,
        )), ValueError,
                 "decode metadata must match the already-audited source FPS")


def test_existing_exports_are_audited_but_never_reused(c: Check):
    with tempfile.TemporaryDirectory() as td:
        data, exports, _inventory_ok, probe, inspect_arkit = _inventory(Path(td))

        def live_like_probe(path: Path):
            metadata = probe(path)
            if path.parent.name == "PHI_existing_00":
                return builder.VideoMetadata(
                    frame_count=metadata.frame_count, fps=59.95,
                    width=metadata.width, height=metadata.height,
                )
            return metadata

        inventory = builder.inventory_mayo_sources(
            data, exports, probe_video=live_like_probe,
            inspect_arkit=inspect_arkit, enforce_frozen=True,
        )
        collection, _exposure = builder.build_public_manifests(
            inventory, b"local-secret-salt-for-mayo-ssl-0123456789"
        )
        rows = collection["mediapipe_records"]
        c.eq(sum(row["legacy_export_audit_status"]
                 == "not_reused_unverifiable_source_binding" for row in rows), 13)
        c.true(all(row["cache_source"]
                   == "raw_video_reextracted_homogeneous_video_mode" for row in rows),
               "all 48 final caches use one producer regardless of legacy availability")
        c.eq(collection["temporal_protocol"]["source_timeline"],
             "per_recording_audited_fps_and_monotonic_source_index")


def _arkit_timecode(source_index: int, subframe: int = 746) -> str:
    absolute = 12 * 3600 * 60 + source_index
    hours, within_hour = divmod(absolute, 3600 * 60)
    minutes, within_minute = divmod(within_hour, 60 * 60)
    seconds, frames = divmod(within_minute, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}.{subframe:03d}"


def _write_arkit_csv(
    path: Path,
    rows: int = 3,
    *,
    source_indices: tuple[int, ...] | None = None,
) -> None:
    rotations = (
        "HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw", "LeftEyePitch",
        "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
    )
    header = ("Timecode", "BlendshapeCount", *builder.ARKIT_BLENDSHAPE_NAMES, *rotations)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        indices = source_indices if source_indices is not None else tuple(range(rows))
        for row, source_index in enumerate(indices):
            writer.writerow((_arkit_timecode(source_index), 61,
                             *[0.01 * (row + index) for index in range(52)],
                             *([0.0] * 9)))


def test_arkit_timecode_preserves_gaps_and_rejects_nonmonotonic_rows(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "private_iPhone.csv"
        _write_arkit_csv(source, source_indices=(0, 1, 3, 4, 6))
        inspection = builder.inspect_arkit_csv(source)
        c.eq(inspection.missing_source_frames, 2)
        sequence = builder.load_arkit_trajectory(source)
        c.eq(sequence.source_frame_indices.tolist(), [0, 1, 3, 4, 6])
        c.eq(sequence.timestamps.tolist(), [0.0, 1 / 60, 3 / 60, 4 / 60, 6 / 60])

        rec = "rec_" + "a" * 64
        cached_path = root / f"{rec}.npz"
        builder.write_arkit_cache(
            cached_path, sequence, recording_id=rec, group_id="grp_" + "b" * 64,
            source_integrity_id="src_" + "c" * 64,
            source_fingerprint="fp_" + "d" * 64,
        )
        with np.load(cached_path, allow_pickle=False) as cached:
            c.eq(cached["source_frame_indices_30hz"].tolist(), [0, 4, 6],
                 "30-Hz ARKit view never nearest-fills missing target indices 2")
            c.eq(cached["target_frame_indices_30hz"].tolist(), [0, 2, 3])
            c.eq(cached["contiguous_from_previous_30hz"].tolist(),
                 [False, False, True])

        duplicate = root / "duplicate_iPhone.csv"
        _write_arkit_csv(duplicate, source_indices=(0, 1, 1))
        c.raises(lambda: builder.load_arkit_trajectory(duplicate), ValueError,
                 "duplicate original Timecode is rejected")
        backward = root / "backward_iPhone.csv"
        _write_arkit_csv(backward, source_indices=(0, 2, 1))
        c.raises(lambda: builder.load_arkit_trajectory(backward), ValueError,
                 "backward original Timecode is rejected")


def test_compact_caches_keep_mediapipe_and_arkit_modalities_separate(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rec = "rec_" + "1" * 64
        group = "grp_" + "2" * 64
        source_hash = "3" * 64
        fingerprint = "fp_" + "4" * 64
        media_path = root / f"{rec}.npz"
        builder.write_mediapipe_cache(
            media_path, _sequence_with_gap(), recording_id=rec, group_id=group,
            source_integrity_id="src_" + source_hash,
            source_fingerprint=fingerprint,
        )
        with np.load(media_path, allow_pickle=False) as cached:
            c.eq(cached["features_source_rate"].shape, (6, 95))
            c.eq(cached["features_30hz"].shape, (3, 95))
            c.eq(float(cached["source_fps"].item()), 60.0)
            c.eq(cached["target_frame_indices_30hz"].tolist(), [0, 1, 3])
            c.true("features_60hz" not in cached.files,
                   "MediaPipe source tensors are never mislabeled as exactly 60 Hz")
            c.eq(str(cached["feature_schema"].item()), DYNAMIC_FEATURE_SCHEMA)
            c.true("annotated_preview" not in cached.files)
            c.true("source_sha256" not in cached.files,
                   "shareable compact caches never persist a raw source digest")

        arkit_csv = root / "private_iPhone.csv"
        _write_arkit_csv(arkit_csv)
        arkit = builder.load_arkit_trajectory(arkit_csv)
        c.eq(arkit.features.shape, (3, 52))
        arkit_rec = "rec_" + "5" * 64
        arkit_path = root / f"{arkit_rec}.npz"
        builder.write_arkit_cache(
            arkit_path, arkit, recording_id=arkit_rec, group_id="grp_" + "6" * 64,
            source_integrity_id="src_" + "7" * 64,
            source_fingerprint="fp_" + "8" * 64,
        )
        with np.load(arkit_path, allow_pickle=False) as cached:
            c.eq(cached["features_60hz"].shape, (3, 52))
            c.eq(str(cached["feature_schema"].item()), "arkit_blendshapes_52_v1")
            c.true(not any("landmark" in name.lower() for name in cached.files),
                   "ARKit-only trajectories never receive fabricated landmark columns")
            c.true("source_sha256" not in cached.files,
                   "ARKit compact caches never persist a raw source digest")
        c.eq(sorted(path.suffix for path in root.iterdir()),
             [".csv", ".npz", ".npz"],
             "cache writing creates no annotated preview or giant derived CSV")


def test_provenance_hashes_dependencies_and_detects_toctou(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source.mov"
        model = root / "face_landmarker.task"
        producer = root / "producer.py"
        python_executable = root / "python"
        source.write_bytes(b"source-v1")
        model.write_bytes(b"model-v1")
        producer.write_text("producer-v1")
        python_executable.write_bytes(b"python-runtime-v1")

        versions = {
            "numpy": "1.2.3", "mediapipe": "0.10.35", "opencv-python": "4.9.0",
        }

        def resolver(name: str) -> str:
            if name not in versions:
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        artifact_paths = {}
        for distribution in versions:
            dist_info = root / f"{distribution}.dist-info"
            dist_info.mkdir()
            metadata = dist_info / "METADATA"
            record = dist_info / "RECORD"
            installed = root / distribution
            installed_bytes = f"installed-{distribution}".encode()
            installed.write_bytes(installed_bytes)
            metadata_bytes = (
                f"Name: {distribution}\nVersion: {versions[distribution]}\n"
            ).encode()
            metadata.write_bytes(metadata_bytes)
            with record.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows((
                    (distribution, _record_digest(installed_bytes),
                     str(len(installed_bytes))),
                    (f"{distribution}.dist-info/METADATA",
                     _record_digest(metadata_bytes), str(len(metadata_bytes))),
                    (f"{distribution}.dist-info/RECORD", "", ""),
                ))
            artifact_paths[distribution] = (metadata, record)

        def artifact_resolver(distribution: str):
            return artifact_paths[distribution]

        snapshot = builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            dependency_artifact_resolver=artifact_resolver,
            python_executable=python_executable,
        )
        c.eq(snapshot.model_sha256, _sha(b"model-v1"))
        c.eq(snapshot.source_sha256, (_sha(b"source-v1"),))
        c.eq(snapshot.producer_sha256["builder"], _sha(b"producer-v1"))
        c.true(snapshot.dependencies["mediapipe"] == "mediapipe==0.10.35")
        c.eq(snapshot.dependency_sha256["python_executable"],
             _sha(b"python-runtime-v1"))
        c.true({"numpy_metadata", "numpy_record", "mediapipe_metadata",
                "mediapipe_record", "opencv_metadata", "opencv_record"}
               .issubset(snapshot.dependency_sha256),
               "dependency provenance fingerprints wheel metadata and RECORD artifacts")
        builder.assert_provenance_unchanged(
            snapshot, version_resolver=resolver,
            dependency_artifact_resolver=artifact_resolver,
            python_executable=python_executable,
        )
        artifact_paths["mediapipe"][0].write_text("changed metadata")
        c.raises(lambda: builder.assert_provenance_unchanged(
            snapshot, version_resolver=resolver,
            dependency_artifact_resolver=artifact_resolver,
            python_executable=python_executable,
        ), ValueError, "dependency artifact drift blocks promotion")
        artifact_paths["mediapipe"][0].write_text(
            "Name: mediapipe\nVersion: 0.10.35\n"
        )
        source.write_bytes(b"source-v2")
        c.raises(lambda: builder.assert_provenance_unchanged(
            snapshot, version_resolver=resolver,
            dependency_artifact_resolver=artifact_resolver,
            python_executable=python_executable,
        ), ValueError,
                 "a source mutation before promotion is detected")
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            expected_source_hashes={source: _sha(b"source-v1")},
            dependency_artifact_resolver=artifact_resolver,
            python_executable=python_executable,
        ), ValueError,
                 "inventory-to-snapshot source replacement cannot establish a new baseline")


def test_dependency_record_closure_hashes_installed_code_and_native_bytes(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        runtime = root / "runtime"
        site = runtime / "lib" / "python3.10" / "site-packages"
        python_executable = runtime / "bin" / "python"
        python_executable.parent.mkdir(parents=True)
        python_executable.write_bytes(b"python-runtime")
        site.mkdir(parents=True)
        versions = {
            "numpy": "1.2.3", "mediapipe": "0.10.35", "opencv-python": "4.9.0",
        }
        artifact_paths = {}
        installed_code = {}
        record_rows = {}
        mutable_pyc = None
        for index, (distribution, version) in enumerate(versions.items()):
            package = site / f"package_{index}"
            package.mkdir()
            suffix = ".so" if distribution == "mediapipe" else ".py"
            code = package / f"runtime{suffix}"
            code_bytes = f"runtime-code-{distribution}".encode("ascii")
            code.write_bytes(code_bytes)
            dist_info = site / f"{distribution}-{version}.dist-info"
            dist_info.mkdir()
            metadata = dist_info / "METADATA"
            metadata_bytes = f"Name: {distribution}\nVersion: {version}\n".encode()
            metadata.write_bytes(metadata_bytes)
            record = dist_info / "RECORD"
            rows = [
                (code.relative_to(site).as_posix(), _record_digest(code_bytes),
                 str(len(code_bytes))),
                (metadata.relative_to(site).as_posix(), _record_digest(metadata_bytes),
                 str(len(metadata_bytes))),
                (record.relative_to(site).as_posix(), "", ""),
            ]
            if distribution == "numpy":
                pycache = package / "__pycache__"
                pycache.mkdir()
                mutable_pyc = pycache / "runtime.cpython-310.pyc"
                pyc_bytes = b"runtime-generated-bytecode"
                mutable_pyc.write_bytes(pyc_bytes)
                pyc_name = mutable_pyc.relative_to(site).as_posix()
                rows[1:1] = [
                    (pyc_name, "", ""),
                    (pyc_name, _record_digest(b"install-time-bytecode"), "21"),
                ]
            with record.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            artifact_paths[distribution] = (metadata, record)
            installed_code[distribution] = (code, code_bytes)
            record_rows[distribution] = rows

        def resolver(name: str) -> str:
            if name not in versions:
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        source = root / "source.mov"
        model = root / "model.task"
        producer = root / "producer.py"
        source.write_bytes(b"source")
        model.write_bytes(b"model")
        producer.write_bytes(b"producer")
        snapshot = builder.snapshot_provenance(
            [source], model, {"builder": producer},
            version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        )
        builder.assert_provenance_unchanged(
            snapshot, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        )
        c.eq(snapshot.dependency_file_counts, {
            "python": 1, "numpy": 4, "mediapipe": 3, "opencv-python": 3,
        }, "runtime provenance exposes installed-file counts without paths")
        code, original = installed_code["mediapipe"]
        code.write_bytes(b"X" * len(original))
        c.raises(lambda: builder.assert_provenance_unchanged(
            snapshot, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError,
                 "installed .py/.so drift is detected even when METADATA and RECORD stay fixed")
        code.write_bytes(original)
        assert mutable_pyc is not None
        mutable_pyc.write_bytes(b"changed-runtime-bytecode")
        c.raises(lambda: builder.assert_provenance_unchanged(
            snapshot, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError,
                 "mutable pyc is still frozen from snapshot through promotion")
        mutable_pyc.write_bytes(b"runtime-generated-bytecode")

        mediapipe_record = artifact_paths["mediapipe"][1]
        ordinary_duplicate = list(record_rows["mediapipe"])
        ordinary_duplicate.insert(1, (ordinary_duplicate[0][0], "", ""))
        with mediapipe_record.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(ordinary_duplicate)
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError,
                 "blank-plus-hashed duplicate semantics are forbidden for .py/.so files")

        def write_mediapipe_rows(rows):
            with mediapipe_record.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)

        missing_rows = list(record_rows["mediapipe"])
        missing_rows.insert(1, ("package_1/missing.so", "", ""))
        write_mediapipe_rows(missing_rows)
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError, "a missing RECORD target fails closed")

        escape = runtime.parent / "outside-runtime.so"
        escape.write_bytes(b"outside")
        traversal = os.path.relpath(escape, site).replace(os.sep, "/")
        traversal_rows = list(record_rows["mediapipe"])
        traversal_rows.insert(1, (traversal, "", ""))
        write_mediapipe_rows(traversal_rows)
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError,
                 "RECORD dot-dot paths are allowed only while confined to the exact runtime")

        link = installed_code["mediapipe"][0].parent / "runtime_alias.so"
        link.symlink_to(installed_code["mediapipe"][0])
        symlink_rows = list(record_rows["mediapipe"])
        symlink_rows.insert(1, (link.relative_to(site).as_posix(), "", ""))
        write_mediapipe_rows(symlink_rows)
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError, "a symlinked RECORD target is rejected")

        alias_rows = list(record_rows["mediapipe"])
        alias_rows.insert(1, ("package_1/subdir/../runtime.so", "", ""))
        write_mediapipe_rows(alias_rows)
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, {"builder": producer}, version_resolver=resolver,
            dependency_artifact_resolver=lambda name: artifact_paths[name],
            python_executable=python_executable,
        ), ValueError,
                 "different RECORD spellings cannot alias one normalized target")


def test_hardlink_snapshot_pins_hashed_bytes_and_detects_source_swap(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source.mov"
        source.write_bytes(b"audited-video-bytes")
        snapshot_dir = root / "snapshots"
        snapshot_dir.mkdir()
        pinned = builder.pin_source_file(
            source, snapshot_dir, "source-000.mov",
            expected_sha256=_sha(b"audited-video-bytes"),
        )
        c.eq(pinned.sha256, _sha(b"audited-video-bytes"))
        c.eq(os.stat(source).st_ino, os.stat(pinned.pinned_path).st_ino,
             "decoder snapshot is a pinned hard link to the exact hashed inode")

        replacement = root / "replacement.mov"
        replacement.write_bytes(b"swapped-video-bytes")
        os.replace(replacement, source)
        c.eq(pinned.pinned_path.read_bytes(), b"audited-video-bytes",
             "the bytes subsequently decoded remain the bytes that were hashed")
        c.raises(lambda: builder.assert_pinned_source_unchanged(pinned), ValueError,
                 "postvalidation detects a source-path inode swap")


def test_extractor_lifecycle_lock_and_transaction_are_fail_closed(c: Check):
    class FakeExtractor:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    created: list[FakeExtractor] = []

    def factory(**kwargs):
        c.eq(set(kwargs), {"model_path"},
             "managed producer cannot silently fall back to IMAGE-mode options")
        item = FakeExtractor()
        created.append(item)
        return item

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model = root / "model.task"
        model.write_bytes(b"model")
        try:
            with builder.managed_extractor(factory, model_path=model):
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        c.eq(len(created), 1, "one extractor instance is constructed")
        c.eq(created[0].closed, 1, "the underlying resource closes exactly once")

        output = root / "cache"
        output.mkdir()
        (output / "sentinel").write_text("old")
        staging = root / ".cache.staging-test"
        staging.mkdir()
        (staging / "sentinel").write_text("new")
        with builder.output_parent_lock(output):
            def nested():
                with builder.output_parent_lock(output):
                    pass
            c.raises(nested, RuntimeError, "same-process concurrent generation is rejected")
            builder.promote_generation(staging, output)
        c.eq((output / "sentinel").read_text(), "new")

        staging2 = root / ".cache.staging-failure"
        staging2.mkdir()
        (staging2 / "sentinel").write_text("bad-new")
        failed = False

        def fail_once(src, dst):
            nonlocal failed
            if Path(src) == staging2 and Path(dst) == output and not failed:
                failed = True
                raise OSError("forced promotion failure")
            return os.replace(src, dst)

        with builder.output_parent_lock(output):
            c.raises(lambda: builder.promote_generation(
                staging2, output, replace_func=fail_once), OSError,
                "failed promotion restores the prior complete generation")
        c.eq((output / "sentinel").read_text(), "new")

        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        output.chmod(0o700)
        (output / "sentinel").chmod(0o600)
        exposure.chmod(0o600)
        staging3 = _canonical_transaction_staging(
            root, ".cache.staging-exposure-failure",
            b"failed-exposure-salt-012345678901",
        )

        def fail_exposure_install(src, dst):
            if Path(dst) == exposure and ".tmp-" in Path(src).name:
                raise OSError("forced exposure install failure")
            return os.replace(src, dst)

        with builder.output_parent_lock(output):
            c.raises(lambda: builder.promote_generation(
                staging3, output, exposure_manifest_path=exposure,
                replace_func=fail_exposure_install), OSError,
                "cache and ignored exposure manifest promote as one transaction")
        c.eq((output / "sentinel").read_text(), "new")
        c.eq(exposure.read_text(), "old-exposure")


def _test_provenance(salt: bytes) -> dict[str, object]:
    return {
        "runtime_dependencies": [
            {
                "distribution": distribution,
                "version": "1.0.0",
                "installed_file_count": 1,
                "installed_file_aggregate_sha256": digest * 64,
            }
            for distribution, digest in (
                ("mediapipe", "1"), ("numpy", "2"),
                ("opencv-python", "3"), ("python", "4"),
            )
        ],
        "dependency_aggregate_sha256": "5" * 64,
        "model_sha256": "6" * 64,
        "source_collection_integrity_id": builder.hmac_identifier(
            "agg", salt, "mayo-source-collection-integrity", "7" * 64
        ),
        "producer_sha256": {
            "action_bundle": "8" * 64,
            "builder": "9" * 64,
            "clinical_landmarks": "a" * 64,
            "dynamic_landmark_schema": "b" * 64,
            "feature_registry": "c" * 64,
        },
        "producer_aggregate_sha256": "d" * 64,
    }


def _semantic_staging(
    root: Path,
    name: str,
    salt: bytes,
    *,
    include_arkit: bool,
    include_exclusions: bool = False,
) -> Path:
    staging = root / name
    media = staging / "mediapipe"
    arkit = staging / "arkit"
    staging.mkdir(mode=0o700)
    media.mkdir(mode=0o700)
    arkit.mkdir(mode=0o700)
    for directory in (staging, media, arkit):
        directory.chmod(0o700)
    private = root / "private_fixture"
    video = builder.VideoAsset(
        private / "video", private / "video" / "source.mov",
        builder.VideoMetadata(6, 60.0, 10, 10), "1" * 64, None,
    )
    duplicate = builder.VideoAsset(
        private / "duplicate", private / "duplicate" / "copy.mov",
        builder.VideoMetadata(6, 60.0, 10, 10), "1" * 64, None,
    )
    short_one = builder.VideoAsset(
        private / "short_one", private / "short_one" / "qc.mov",
        builder.VideoMetadata(30, 60.0, 10, 10), "3" * 64, None,
    )
    short_two = builder.VideoAsset(
        private / "short_two", private / "short_two" / "qc.mov",
        builder.VideoMetadata(45, 60.0, 10, 10), "4" * 64, None,
    )
    arkit_asset = builder.ARKitAsset(
        private / "arkit", private / "arkit" / "source_iPhone.csv",
        5, builder.ARKIT_BLENDSHAPE_NAMES, "2" * 64, 2,
    )
    counts = {
        "total_sessions": 1 + int(include_arkit) + 3 * int(include_exclusions),
        "video_bearing_sessions": 1 + 3 * int(include_exclusions),
        "without_video_sessions": int(include_arkit),
        "exact_duplicate_copies_excluded": int(include_exclusions),
        "short_qc_clips_excluded": 2 * int(include_exclusions),
        "long_unique_videos": 1,
        "existing_complete_v2_exports": 0,
        "remaining_long_videos": 1,
        "remaining_long_video_frames": 6,
        "arkit_only_sessions": int(include_arkit),
        "arkit_trajectories": int(include_arkit),
        "arkit_rows": 5 if include_arkit else 0,
        "arkit_timecode_gaps": 2 if include_arkit else 0,
        "metadata_only_sessions": 0,
    }
    inventory = builder.MayoInventory(
        private, private / "exports", counts,
        ((video, duplicate, short_one, short_two)
         if include_exclusions else (video,)),
        (video,), (), (video,),
        ((duplicate,) if include_exclusions else ()),
        ((short_one, short_two) if include_exclusions else ()),
        ((private / "arkit",) if include_arkit else ()),
        ((arkit_asset,) if include_arkit else ()), (),
    )
    collection, exposure = builder.build_public_manifests(inventory, salt)
    collection["provenance"] = _test_provenance(salt)
    media_row = collection["mediapipe_records"][0]
    media_path = media / f"{media_row['recording_id']}.npz"
    builder.write_mediapipe_cache(
        media_path, _sequence_with_gap(),
        recording_id=media_row["recording_id"], group_id=media_row["group_id"],
        source_integrity_id=media_row["source_integrity_id"],
        source_fingerprint=media_row["source_fingerprint"],
    )
    media_integrity = builder.hmac_identifier(
        "cache", salt, "mayo-mediapipe-cache-integrity",
        _sha(media_path.read_bytes()),
    )
    media_row["cache_integrity_id"] = media_integrity
    next(row for row in exposure["videos"]
         if row["status"] == "mediapipe_ssl")["cache_integrity_id"] = media_integrity
    if include_arkit:
        arkit_row = collection["arkit_records"][0]
        source_indices = np.asarray([0, 1, 3, 4, 6], dtype=np.int64)
        arkit_sequence = builder.ARKitSequence(
            features=np.repeat(source_indices[:, None], 52, axis=1).astype(np.float32),
            valid_mask=np.ones(5, dtype=bool),
            timestamps=source_indices.astype(np.float64) / 60.0,
            source_frame_indices=source_indices,
        )
        arkit_path = arkit / f"{arkit_row['recording_id']}.npz"
        builder.write_arkit_cache(
            arkit_path, arkit_sequence,
            recording_id=arkit_row["recording_id"], group_id=arkit_row["group_id"],
            source_integrity_id=arkit_row["source_integrity_id"],
            source_fingerprint=arkit_row["source_fingerprint"],
        )
        arkit_integrity = builder.hmac_identifier(
            "cache", salt, "mayo-arkit-cache-integrity",
            _sha(arkit_path.read_bytes()),
        )
        arkit_row["cache_integrity_id"] = arkit_integrity
    _write_staging_manifests(
        staging / "collection_manifest.json",
        staging / "mayo_exposure_manifest.json",
        collection,
        exposure,
    )
    return staging


def _canonical_transaction_staging(root: Path, name: str, salt: bytes) -> Path:
    return _semantic_staging(root, name, salt, include_arkit=False)


def _media_sequence_of_length(length: int) -> object:
    indices = np.arange(length, dtype=np.int64)
    return builder.MayoMediaSequence(
        features=np.repeat(indices[:, None], 95, axis=1).astype(np.float32),
        valid_mask=np.ones(length, dtype=bool),
        timestamps=indices.astype(np.float64) / 60.0,
        source_frame_indices=indices,
        facial_transforms=np.repeat(
            np.eye(4, dtype=np.float32)[None], length, axis=0
        ),
        facial_transform_mask=np.ones(length, dtype=bool),
        transform_source="same_detection_mediapipe_video_mode",
    )


def _legacy_swap_staging(root: Path, name: str, salt: bytes) -> Path:
    staging = root / name
    media = staging / "mediapipe"
    arkit = staging / "arkit"
    staging.mkdir(mode=0o700)
    media.mkdir(mode=0o700)
    arkit.mkdir(mode=0o700)
    for directory in (staging, media, arkit):
        directory.chmod(0o700)
    private = root / "private_legacy_fixture"
    lengths = (2, 5, 3, 4)
    assets = tuple(
        builder.VideoAsset(
            private / f"video_{index}",
            private / f"video_{index}" / "source.mov",
            builder.VideoMetadata(length, 60.0, 10, 10),
            f"{index + 1:x}" * 64,
            None,
        )
        for index, length in enumerate(lengths)
    )
    counts = {
        "total_sessions": 4,
        "video_bearing_sessions": 4,
        "without_video_sessions": 0,
        "exact_duplicate_copies_excluded": 0,
        "short_qc_clips_excluded": 0,
        "long_unique_videos": 4,
        "existing_complete_v2_exports": 2,
        "remaining_long_videos": 2,
        "remaining_long_video_frames": 7,
        "arkit_only_sessions": 0,
        "arkit_trajectories": 0,
        "arkit_rows": 0,
        "arkit_timecode_gaps": 0,
        "metadata_only_sessions": 0,
    }
    inventory = builder.MayoInventory(
        private, private / "exports", counts,
        assets, assets, assets[:2], assets[2:], (), (), (), (), (),
    )
    collection, exposure = builder.build_public_manifests(inventory, salt)
    collection["provenance"] = _test_provenance(salt)
    exposure_by_recording = {
        row["recording_id"]: row for row in exposure["videos"]
    }
    rows_by_recording = {
        row["recording_id"]: row for row in collection["mediapipe_records"]
    }
    for asset, length in zip(assets, lengths):
        recording_id = builder.hmac_identifier(
            "rec", salt, "mayo-mediapipe-recording", asset.source_sha256
        )
        row = rows_by_recording[recording_id]
        cache_path = media / f"{recording_id}.npz"
        builder.write_mediapipe_cache(
            cache_path, _media_sequence_of_length(length),
            recording_id=recording_id, group_id=row["group_id"],
            source_integrity_id=row["source_integrity_id"],
            source_fingerprint=row["source_fingerprint"],
        )
        integrity = builder.hmac_identifier(
            "cache", salt, "mayo-mediapipe-cache-integrity",
            _sha(cache_path.read_bytes()),
        )
        row["cache_integrity_id"] = integrity
        exposure_by_recording[recording_id]["cache_integrity_id"] = integrity
    _write_staging_manifests(
        staging / "collection_manifest.json",
        staging / "mayo_exposure_manifest.json",
        collection, exposure,
    )
    return staging


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: loaded[name].copy() for name in loaded.files}
    mutate(payload)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def _patch_cache_central_size(path: Path, *, field_offset: int) -> None:
    payload = bytearray(path.read_bytes())
    central = payload.index(b"PK\x01\x02")
    payload[central + field_offset:central + field_offset + 4] = (
        0x7FFF_FFFF
    ).to_bytes(4, "little")
    path.write_bytes(payload)


def _append_unexpected_npz_member(path: Path) -> None:
    member = io.BytesIO()
    np.save(member, np.asarray(1, dtype=np.int64), allow_pickle=False)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("unexpected.npy", member.getvalue())


def _refresh_cache_integrity(staging: Path, salt: bytes, modality: str) -> None:
    collection_path = staging / "collection_manifest.json"
    exposure_path = staging / "mayo_exposure_manifest.json"
    collection = json.loads(collection_path.read_text())
    exposure = json.loads(exposure_path.read_text())
    manifest_key = "mediapipe_records" if modality == "mediapipe" else "arkit_records"
    row = collection[manifest_key][0]
    cache_path = staging / modality / f"{row['recording_id']}.npz"
    context = (
        "mayo-mediapipe-cache-integrity"
        if modality == "mediapipe" else "mayo-arkit-cache-integrity"
    )
    integrity = builder.hmac_identifier(
        "cache", salt, context, _sha(cache_path.read_bytes())
    )
    row["cache_integrity_id"] = integrity
    if modality == "mediapipe":
        exposed = next(
            item for item in exposure["videos"]
            if item["status"] == "mediapipe_ssl"
        )
        exposed["cache_integrity_id"] = integrity
    else:
        exposure["arkit_trajectories"][0]["cache_integrity_id"] = integrity
    collection_path.write_text(json.dumps(collection, sort_keys=True))
    exposure_path.write_text(json.dumps(exposure, sort_keys=True))


def _set_nonfinite(payload: dict[str, np.ndarray], field: str) -> None:
    value = payload[field].copy()
    value.flat[0] = np.nan
    payload[field] = value


def _exception_chain_contains(
    error: BaseException,
    exception_type: type[BaseException],
    message: str,
) -> bool:
    observed: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in observed:
        observed.add(id(current))
        if isinstance(current, exception_type) and message in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _duplicate_first_source_index(
    payload: dict[str, np.ndarray], field: str,
) -> None:
    value = payload[field].copy()
    value[1] = value[0]
    payload[field] = value


def _make_media_all_masked(payload: dict[str, np.ndarray]) -> None:
    payload["valid_mask_source_rate"][:] = False
    payload["features_source_rate"][:] = 0.0
    payload["facial_transform_mask_source_rate"][:] = False
    payload["facial_transforms_source_rate"][:] = 0.0


def test_compact_cache_validation_is_exact_and_recomputes_30hz(c: Check):
    media_mutations = (
        ("extra PHI field", lambda p: p.__setitem__(
            "raw_patient_name", np.asarray("PHI_person"))),
        ("missing source features", lambda p: p.pop("features_source_rate")),
        ("wrong feature shape", lambda p: p.__setitem__(
            "features_source_rate", p["features_source_rate"][:, :-1])),
        ("wrong feature dtype", lambda p: p.__setitem__(
            "features_source_rate", p["features_source_rate"].astype(np.float64))),
        ("non-finite features", lambda p: _set_nonfinite(
            p, "features_source_rate")),
        ("wrong source timeline", lambda p: p.__setitem__(
            "timestamps_source_rate", p["timestamps_source_rate"] + 0.001)),
        ("duplicate source index", lambda p: _duplicate_first_source_index(
            p, "source_frame_indices_source_rate")),
        ("all-masked source", _make_media_all_masked),
        ("wrong feature registry", lambda p: p.__setitem__(
            "feature_names", p["feature_names"][::-1])),
        ("wrong schema metadata", lambda p: p.__setitem__(
            "feature_schema", np.asarray("clinical23_v1"))),
        ("wrong scalar governance dtype", lambda p: p.__setitem__(
            "development_only", np.asarray(1, dtype=np.int64))),
        ("independent 30-Hz features", lambda p: p.__setitem__(
            "features_30hz", p["features_30hz"] + np.float32(0.25))),
    )
    arkit_mutations = (
        ("extra PHI field", lambda p: p.__setitem__(
            "raw_patient_name", np.asarray("PHI_person"))),
        ("missing source features", lambda p: p.pop("features_60hz")),
        ("wrong feature shape", lambda p: p.__setitem__(
            "features_60hz", p["features_60hz"][:, :-1])),
        ("wrong feature dtype", lambda p: p.__setitem__(
            "features_60hz", p["features_60hz"].astype(np.float64))),
        ("non-finite features", lambda p: _set_nonfinite(p, "features_60hz")),
        ("wrong Timecode timeline", lambda p: p.__setitem__(
            "timestamps_60hz", p["timestamps_60hz"] + 0.001)),
        ("invalid Timecode gap", lambda p: _duplicate_first_source_index(
            p, "source_frame_indices_60hz")),
        ("invalid mask", lambda p: p.__setitem__(
            "valid_mask_60hz", np.zeros_like(p["valid_mask_60hz"]))),
        ("wrong feature registry", lambda p: p.__setitem__(
            "feature_names", p["feature_names"][::-1])),
        ("wrong schema metadata", lambda p: p.__setitem__(
            "feature_schema", np.asarray("arkit_blendshapes_unknown"))),
        ("wrong scalar governance dtype", lambda p: p.__setitem__(
            "development_only", np.asarray(1, dtype=np.int64))),
        ("independent 30-Hz indices", lambda p: p.__setitem__(
            "target_frame_indices_30hz", p["target_frame_indices_30hz"] + 1)),
    )
    salt = b"exact-cache-validation-salt-012345678"
    for modality, mutations in (
        ("mediapipe", media_mutations), ("arkit", arkit_mutations)
    ):
        for index, (label, mutate) in enumerate(mutations):
            with tempfile.TemporaryDirectory() as td:
                staging = _semantic_staging(
                    Path(td), f".cache.staging-{modality}-{index}", salt,
                    include_arkit=True,
                )
                cache_path = next((staging / modality).glob("*.npz"))
                _rewrite_npz(cache_path, mutate)
                _refresh_cache_integrity(staging, salt, modality)
                c.raises(
                    lambda: builder._validate_staging(staging, salt=salt),
                    ValueError,
                    f"{modality} {label} fails despite a freshly valid cache HMAC/tree",
                )


def test_mayo_npz_resource_metadata_is_rejected_before_numpy_load(c: Check):
    salt = b"bounded-npz-metadata-salt-0123456789"
    attacks = (
        ("declared compressed bytes",
         lambda path: _patch_cache_central_size(path, field_offset=20)),
        ("declared expanded bytes",
         lambda path: _patch_cache_central_size(path, field_offset=24)),
        ("excessive member count", _append_unexpected_npz_member),
    )
    for index, (label, mutate) in enumerate(attacks):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-bounded-{index}", salt,
                include_arkit=False,
            )
            cache_path = next((staging / "mediapipe").glob("*.npz"))
            mutate(cache_path)
            _refresh_cache_integrity(staging, salt, "mediapipe")
            original_load = builder.np.load

            def materialized_too_early(*_args, **_kwargs):
                raise RuntimeError("np.load reached before bounded ZIP inspection")

            builder.np.load = materialized_too_early
            try:
                c.raises(
                    lambda: builder._validate_staging(staging, salt=salt),
                    ValueError,
                    f"Mayo {label} is rejected before NumPy materialization",
                )
            finally:
                builder.np.load = original_load


def test_mayo_zip_preflight_counts_actual_central_records_before_zipfile(c: Check):
    buffer = io.BytesIO()
    actual_record_count = len(builder._MEDIAPIPE_CACHE_FIELDS) + 96
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(actual_record_count):
            archive.writestr(f"member-{index:04d}.npy", b"")
    payload = bytearray(buffer.getvalue())
    eocd = payload.rfind(b"PK\x05\x06")
    c.true(eocd >= 0, "the fixture has one ordinary EOCD")
    claimed = len(builder._MEDIAPIPE_CACHE_FIELDS)
    payload[eocd + 8:eocd + 10] = claimed.to_bytes(2, "little")
    payload[eocd + 10:eocd + 12] = claimed.to_bytes(2, "little")

    original_zipfile = builder.zipfile.ZipFile
    materialization_calls = 0

    def materialized_too_early(*_args, **_kwargs):
        nonlocal materialization_calls
        materialization_calls += 1
        raise RuntimeError("ZipFile reached before bounded central-directory parsing")

    builder.zipfile.ZipFile = materialized_too_early
    try:
        c.raises(lambda: builder._require_mayo_npz_headers(
            bytes(payload),
            recording_id="rec_" + "1" * 64,
            group_id="grp_" + "2" * 64,
            source_integrity_id="src_" + "3" * 64,
            source_fingerprint="fp_" + "4" * 64,
            expected_schema=builder.MEDIAPIPE_CACHE_SCHEMA,
        ), ValueError,
        "actual central-directory record count is rejected before ZipFile")
    finally:
        builder.zipfile.ZipFile = original_zipfile
    c.eq(materialization_calls, 0,
         "preflight never allocates ZipInfo records from a forged EOCD count")


def test_mayo_npy_dtype_and_shape_are_rejected_before_numpy_load(c: Check):
    salt = b"bounded-npy-header-salt-01234567890"
    attacks = (
        ("wrong feature dtype", lambda payload: payload.__setitem__(
            "features_source_rate",
            payload["features_source_rate"].astype(np.float64),
        )),
        ("wrong feature shape", lambda payload: payload.__setitem__(
            "features_source_rate", payload["features_source_rate"][:, :-1],
        )),
    )
    for index, (label, mutate) in enumerate(attacks):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-header-{index}", salt,
                include_arkit=False,
            )
            cache_path = next((staging / "mediapipe").glob("*.npz"))
            _rewrite_npz(cache_path, mutate)
            _refresh_cache_integrity(staging, salt, "mediapipe")
            original_load = builder.np.load

            def materialized_too_early(*_args, **_kwargs):
                raise RuntimeError("np.load reached before NPY header validation")

            builder.np.load = materialized_too_early
            try:
                c.raises(
                    lambda: builder._validate_staging(staging, salt=salt),
                    ValueError,
                    f"Mayo {label} is rejected from its NPY header",
                )
            finally:
                builder.np.load = original_load


def test_mayo_raw_cache_limit_is_checked_before_read(c: Check):
    with tempfile.TemporaryDirectory() as td:
        Path(td).chmod(0o700)
        path = Path(td) / "oversized.npz"
        path.write_bytes(b"x" * 65)
        path.chmod(0o600)
        original_read = builder.os.read

        def read_must_not_run(*_args, **_kwargs):
            raise RuntimeError("oversized raw file was read")

        builder.os.read = read_must_not_run
        try:
            c.raises(
                lambda: builder._read_regular_bytes(
                    path, "compact cache", max_bytes=64
                ),
                ValueError,
                "Mayo raw cache size is gated from fstat before reading",
            )
        finally:
            builder.os.read = original_read


def test_mayo_staging_checks_raw_hmac_before_numpy_load(c: Check):
    salt = b"raw-hmac-before-numpy-salt-012345678"
    with tempfile.TemporaryDirectory() as td:
        staging = _semantic_staging(
            Path(td), ".cache.staging-hmac-order", salt, include_arkit=False
        )
        cache_path = next((staging / "mediapipe").glob("*.npz"))
        cache_path.write_bytes(cache_path.read_bytes() + b"tampered-after-eocd")
        original_load = builder.np.load
        load_calls = 0

        def materialized_too_early(*_args, **_kwargs):
            nonlocal load_calls
            load_calls += 1
            raise RuntimeError("np.load reached before raw cache HMAC")

        builder.np.load = materialized_too_early
        try:
            c.raises(
                lambda: builder._validate_staging(staging, salt=salt),
                ValueError,
                "a raw-byte HMAC mismatch is rejected before NumPy materialization",
            )
        finally:
            builder.np.load = original_load
        c.eq(load_calls, 0, "HMAC rejection decompresses no NPY member")


def test_mayo_staging_reads_npz_members_from_held_directory_fd(c: Check):
    salt = b"anchored-cache-read-salt-012345678901"
    with tempfile.TemporaryDirectory() as td:
        staging = _semantic_staging(
            Path(td), ".cache.staging-anchored-read", salt, include_arkit=True
        )
        original_read = builder._read_regular_bytes
        cache_parent_descriptors: list[int | None] = []

        def tracked_read(path, field, **kwargs):
            if field == "compact cache":
                cache_parent_descriptors.append(kwargs.get("parent_descriptor"))
            return original_read(path, field, **kwargs)

        builder._read_regular_bytes = tracked_read
        try:
            builder._validate_staging(staging, salt=salt)
        finally:
            builder._read_regular_bytes = original_read
        c.eq(len(cache_parent_descriptors), 2, "both modality caches were read")
        c.true(
            all(isinstance(item, int) for item in cache_parent_descriptors),
            "every staged NPZ read is anchored to its held modality directory FD",
        )


def test_staging_validation_joins_every_identity_and_governance_field(c: Check):
    salt = b"full-manifest-join-salt-0123456789012"
    npz_mutations = (
        ("recording_id", np.asarray("rec_" + "f" * 64)),
        ("group_id", np.asarray("grp_" + "e" * 64)),
        ("source_fingerprint", np.asarray("fp_" + "d" * 64)),
        ("development_only", np.asarray(False)),
        ("patient_identity", np.asarray("known")),
        ("split_unit", np.asarray("patient")),
    )
    for modality in ("mediapipe", "arkit"):
        for index, (field, replacement) in enumerate(npz_mutations):
            with tempfile.TemporaryDirectory() as td:
                staging = _semantic_staging(
                    Path(td), f".cache.staging-{modality}-npz-{index}", salt,
                    include_arkit=True,
                )
                cache_path = next((staging / modality).glob("*.npz"))
                _rewrite_npz(
                    cache_path,
                    lambda payload, key=field, value=replacement:
                    payload.__setitem__(key, value),
                )
                _refresh_cache_integrity(staging, salt, modality)
                c.raises(
                    lambda: builder._validate_staging(staging, salt=salt),
                    ValueError,
                    f"{modality} NPZ {field} cannot escape the manifest join",
                )

    manifest_mutations = (
        ("collection media extra field", "collection", "mediapipe_records",
         "raw_patient_name", "private"),
        ("collection media group", "collection", "mediapipe_records", "group_id",
         "grp_" + "a" * 64),
        ("collection media fingerprint", "collection", "mediapipe_records",
         "source_fingerprint", "fp_" + "b" * 64),
        ("collection media development", "collection", "mediapipe_records",
         "development_only", False),
        ("collection media split", "collection", "mediapipe_records",
         "split_unit", "patient_held_out"),
        ("exposure media group", "exposure", "videos", "group_id",
         "grp_" + "c" * 64),
        ("exposure media fingerprint", "exposure", "videos", "source_fingerprint",
         "fp_" + "d" * 64),
        ("exposure media ssl flag", "exposure", "videos", "ssl_exposed", False),
        ("exposure media identity", "exposure", "videos", "identity_status",
         "known_patient_identity"),
        ("exposure media split", "exposure", "videos", "split_unit",
         "patient_held_out"),
        ("exposure media independent flag", "exposure", "videos",
         "independent_evaluation_eligible", True),
        ("collection ARKit extra field", "collection", "arkit_records",
         "raw_patient_name", "private"),
        ("collection ARKit group", "collection", "arkit_records", "group_id",
         "grp_" + "e" * 64),
        ("exposure ARKit fingerprint", "exposure", "arkit_trajectories",
         "source_fingerprint", "fp_" + "f" * 64),
        ("exposure ARKit development", "exposure", "arkit_trajectories",
         "development_only", False),
    )
    for index, (label, target, list_key, field, replacement) in enumerate(
        manifest_mutations
    ):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-manifest-{index}", salt,
                include_arkit=True,
            )
            collection_path = staging / "collection_manifest.json"
            exposure_path = staging / "mayo_exposure_manifest.json"
            collection = json.loads(collection_path.read_text())
            exposure = json.loads(exposure_path.read_text())
            manifest = collection if target == "collection" else exposure
            manifest[list_key][0][field] = replacement
            collection_path.write_text(json.dumps(collection, sort_keys=True))
            exposure_path.write_text(json.dumps(exposure, sort_keys=True))
            c.raises(
                lambda: builder._validate_staging(staging, salt=salt),
                ValueError, f"{label} cannot escape the collection/exposure join",
            )

    for index, mutate in enumerate((
        lambda collection, _exposure: collection["counts"].__setitem__(
            "long_unique_videos", 2),
        lambda _collection, exposure: exposure.__setitem__("policy", "reusable test"),
        lambda _collection, exposure: exposure["counts"].__setitem__("videos", 2),
    )):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-top-{index}", salt,
                include_arkit=True,
            )
            collection_path = staging / "collection_manifest.json"
            exposure_path = staging / "mayo_exposure_manifest.json"
            collection = json.loads(collection_path.read_text())
            exposure = json.loads(exposure_path.read_text())
            mutate(collection, exposure)
            collection_path.write_text(json.dumps(collection, sort_keys=True))
            exposure_path.write_text(json.dumps(exposure, sort_keys=True))
            c.raises(
                lambda: builder._validate_staging(staging, salt=salt),
                ValueError, "top-level collection/exposure policy is exact",
            )


def _load_staging_manifests(staging: Path):
    collection_path = staging / "collection_manifest.json"
    exposure_path = staging / "mayo_exposure_manifest.json"
    return (
        collection_path,
        exposure_path,
        json.loads(collection_path.read_text()),
        json.loads(exposure_path.read_text()),
    )


def _write_staging_manifests(
    collection_path: Path,
    exposure_path: Path,
    collection: dict[str, object],
    exposure: dict[str, object],
) -> None:
    collection_path.write_text(json.dumps(collection, sort_keys=True))
    exposure_path.write_text(json.dumps(exposure, sort_keys=True))
    collection_path.chmod(0o600)
    exposure_path.chmod(0o600)


def _append_valid_media_cache_row(staging: Path, salt: bytes) -> None:
    collection_path, exposure_path, collection, exposure = _load_staging_manifests(
        staging
    )
    row = collection["mediapipe_records"][0]
    path = staging / "mediapipe" / f"{row['recording_id']}.npz"
    base = _sequence_with_gap()
    indices = np.append(base.source_frame_indices, np.int64(8))
    features = np.concatenate((base.features, np.full((1, 95), 8, np.float32)))
    valid = np.append(base.valid_mask, True)
    transforms = np.concatenate((
        base.facial_transforms, np.eye(4, dtype=np.float32)[None],
    ))
    transform_mask = np.append(base.facial_transform_mask, True)
    builder.write_mediapipe_cache(
        path,
        builder.MayoMediaSequence(
            features=features,
            valid_mask=valid,
            timestamps=indices.astype(np.float64) / 60.0,
            source_frame_indices=indices,
            facial_transforms=transforms,
            facial_transform_mask=transform_mask,
            transform_source="same_detection_mediapipe_video_mode",
        ),
        recording_id=row["recording_id"], group_id=row["group_id"],
        source_integrity_id=row["source_integrity_id"],
        source_fingerprint=row["source_fingerprint"],
    )
    collection["counts"]["remaining_long_video_frames"] = 7
    _write_staging_manifests(collection_path, exposure_path, collection, exposure)
    _refresh_cache_integrity(staging, salt, "mediapipe")


def _append_valid_arkit_cache_row(staging: Path, salt: bytes) -> None:
    collection_path, exposure_path, collection, exposure = _load_staging_manifests(
        staging
    )
    row = collection["arkit_records"][0]
    path = staging / "arkit" / f"{row['recording_id']}.npz"
    with np.load(path, allow_pickle=False) as cached:
        features = np.concatenate((
            cached["features_60hz"], np.full((1, 52), 8, np.float32),
        ))
        indices = np.append(cached["source_frame_indices_60hz"], np.int64(8))
    builder.write_arkit_cache(
        path,
        builder.ARKitSequence(
            features=features,
            valid_mask=np.ones(len(indices), dtype=bool),
            timestamps=indices.astype(np.float64) / 60.0,
            source_frame_indices=indices,
        ),
        recording_id=row["recording_id"], group_id=row["group_id"],
        source_integrity_id=row["source_integrity_id"],
        source_fingerprint=row["source_fingerprint"],
    )
    collection["counts"]["arkit_rows"] = 6
    collection["counts"]["arkit_timecode_gaps"] = 3
    _write_staging_manifests(collection_path, exposure_path, collection, exposure)
    _refresh_cache_integrity(staging, salt, "arkit")


def test_collection_summary_closes_over_private_caches_and_frozen_inventory(c: Check):
    salt = b"collection-summary-closure-salt-0123456"
    for index, field in enumerate((
        "remaining_long_video_frames", "arkit_rows", "arkit_timecode_gaps",
    )):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-count-{index}", salt,
                include_arkit=True,
            )
            collection_path, exposure_path, collection, exposure = (
                _load_staging_manifests(staging)
            )
            collection["counts"][field] += 1
            _write_staging_manifests(
                collection_path, exposure_path, collection, exposure
            )
            c.raises(
                lambda: builder._validate_staging(staging, salt=salt),
                ValueError, f"summary field {field} closes over private cache rows",
            )

    for modality, mutate in (
        ("mediapipe", _append_valid_media_cache_row),
        ("arkit", _append_valid_arkit_cache_row),
    ):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-coordinated-{modality}", salt,
                include_arkit=True,
            )
            original_counts = json.loads(
                (staging / "collection_manifest.json").read_text()
            )["counts"]
            mutate(staging, salt)
            c.raises(
                lambda: builder._validate_staging(
                    staging, salt=salt,
                    expected_inventory_counts=original_counts,
                ),
                ValueError,
                f"coordinated {modality} cache/HMAC/count rewrite breaks frozen inventory",
            )

    with tempfile.TemporaryDirectory() as td:
        staging = _semantic_staging(
            Path(td), ".cache.staging-session-counts", salt,
            include_arkit=True,
        )
        collection_path, exposure_path, collection, exposure = (
            _load_staging_manifests(staging)
        )
        original_counts = dict(collection["counts"])
        collection["counts"]["total_sessions"] += 1
        collection["counts"]["without_video_sessions"] += 1
        collection["counts"]["metadata_only_sessions"] += 1
        collection["metadata_only_exclusions"][
            "index_or_depth_metadata_only_no_video_or_arkit_trajectory"
        ] += 1
        _write_staging_manifests(
            collection_path, exposure_path, collection, exposure
        )
        c.raises(
            lambda: builder._validate_staging(
                staging, salt=salt,
                expected_inventory_counts=original_counts,
            ),
            ValueError,
            "coordinated session-class rewrite is bound to the caller inventory",
        )


def test_short_exposure_identities_are_disjoint_and_classification_is_frozen(c: Check):
    salt = b"short-exposure-identity-salt-01234567"
    identity_fields = (
        "recording_id", "group_id", "source_integrity_id", "source_fingerprint",
    )

    def refresh_classification(exposure):
        exposure["classification_integrity_id"] = (
            builder.exposure_classification_integrity_id(exposure["videos"], salt)
        )

    for scenario in ("retained_copy", "short_collision"):
        with tempfile.TemporaryDirectory() as td:
            staging = _semantic_staging(
                Path(td), f".cache.staging-short-{scenario}", salt,
                include_arkit=False, include_exclusions=True,
            )
            collection_path, exposure_path, collection, exposure = (
                _load_staging_manifests(staging)
            )
            retained = next(
                row for row in exposure["videos"] if row["status"] == "mediapipe_ssl"
            )
            short_rows = [
                row for row in exposure["videos"]
                if row["status"] == "qc_only_short_clip_excluded"
            ]
            source = retained if scenario == "retained_copy" else short_rows[0]
            target = short_rows[0] if scenario == "retained_copy" else short_rows[1]
            for field in identity_fields:
                target[field] = source[field]
            refresh_classification(exposure)
            _write_staging_manifests(
                collection_path, exposure_path, collection, exposure
            )
            c.raises(
                lambda: builder._validate_staging(staging, salt=salt),
                ValueError, f"short identity collision {scenario} fails closed",
            )

    with tempfile.TemporaryDirectory() as td:
        staging = _semantic_staging(
            Path(td), ".cache.staging-status-swap", salt,
            include_arkit=False, include_exclusions=True,
        )
        collection_path, exposure_path, collection, exposure = (
            _load_staging_manifests(staging)
        )
        expected_classification = exposure["classification_integrity_id"]
        retained = next(
            row for row in exposure["videos"] if row["status"] == "mediapipe_ssl"
        )
        short = next(
            row for row in exposure["videos"]
            if row["status"] == "qc_only_short_clip_excluded"
        )
        retained_identity = {field: retained[field] for field in identity_fields}
        short_identity = {field: short[field] for field in identity_fields}
        cache_integrity = retained.pop("cache_integrity_id")
        retained["status"] = "qc_only_short_clip_excluded"
        retained.update(short_identity)
        short["status"] = "mediapipe_ssl"
        short.update(retained_identity)
        short["cache_integrity_id"] = cache_integrity
        refresh_classification(exposure)
        _write_staging_manifests(
            collection_path, exposure_path, collection, exposure
        )
        c.raises(
            lambda: builder._validate_staging(
                staging, salt=salt,
                expected_classification_integrity_id=expected_classification,
            ),
            ValueError,
            "coordinated status/identity/HMAC rewrite breaks frozen classification",
        )


def test_committed_recovery_rechecks_caller_inventory_commitment(c: Check):
    class SimulatedProcessDeath(BaseException):
        pass

    salt = b"committed-inventory-binding-salt-012345"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging = _semantic_staging(
            root, ".cache.staging-expected-inventory", salt,
            include_arkit=True,
        )
        collection = json.loads(
            (staging / "collection_manifest.json").read_text()
        )
        exposure_payload = json.loads(
            (staging / "mayo_exposure_manifest.json").read_text()
        )
        expected_counts = dict(collection["counts"])
        expected_classification = exposure_payload["classification_integrity_id"]
        output = root / "cache"
        exposure = root / "mayo_exposure_manifest.json"

        def interrupt(phase):
            if phase == "committed":
                raise SimulatedProcessDeath(phase)

        try:
            with builder.output_parent_lock(output):
                builder.promote_generation(
                    staging, output, exposure_manifest_path=exposure,
                    phase_hook=interrupt,
                    expected_inventory_counts=expected_counts,
                    expected_classification_integrity_id=expected_classification,
                )
        except SimulatedProcessDeath:
            pass
        else:
            raise AssertionError("committed interruption did not occur")

        collection_path = output / "collection_manifest.json"
        changed = json.loads(collection_path.read_text())
        changed["counts"]["total_sessions"] += 1
        changed["counts"]["without_video_sessions"] += 1
        changed["counts"]["metadata_only_sessions"] += 1
        changed["metadata_only_exclusions"][
            "index_or_depth_metadata_only_no_video_or_arkit_trajectory"
        ] += 1
        collection_path.write_text(json.dumps(changed, sort_keys=True))
        forged_commitment = builder._validate_staging(output, salt=salt)
        journal_path = root / ".cache.transaction.json"
        journal = json.loads(journal_path.read_text())
        journal["generation_commitment"] = forged_commitment
        journal_path.write_text(json.dumps(journal, sort_keys=True))
        journal_path.chmod(0o600)
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure,
                    expected_inventory_counts=expected_counts,
                    expected_classification_integrity_id=expected_classification,
                ),
                ValueError,
                "committed recovery rejects a self-consistent forged count/journal",
            )


def _swap_equal_total_legacy_classes(staging: Path, salt: bytes) -> None:
    collection_path, exposure_path, collection, exposure = _load_staging_manifests(
        staging
    )
    for row in collection["mediapipe_records"]:
        status = row["legacy_export_audit_status"]
        row["legacy_export_audit_status"] = (
            "no_complete_legacy_export"
            if status == "not_reused_unverifiable_source_binding"
            else "not_reused_unverifiable_source_binding"
        )
    collection["classification_integrity_id"] = (
        builder.collection_classification_integrity_id(
            collection["mediapipe_records"], salt
        )
    )
    _write_staging_manifests(
        collection_path, exposure_path, collection, exposure
    )


def test_collection_classification_blocks_equal_total_legacy_status_swaps(c: Check):
    class SimulatedProcessDeath(BaseException):
        pass

    salt = b"collection-classification-salt-01234567"
    with tempfile.TemporaryDirectory() as td:
        staging = _legacy_swap_staging(
            Path(td), ".cache.staging-legacy-swap", salt
        )
        collection = json.loads(
            (staging / "collection_manifest.json").read_text()
        )
        expected_counts = dict(collection["counts"])
        expected_collection_classification = collection[
            "classification_integrity_id"
        ]
        _swap_equal_total_legacy_classes(staging, salt)
        c.raises(
            lambda: builder._validate_staging(
                staging, salt=salt,
                expected_inventory_counts=expected_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification
                ),
            ),
            ValueError,
            "equal-count/equal-frame legacy status swap breaks frozen classification",
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging = _legacy_swap_staging(
            root, ".cache.staging-legacy-committed", salt
        )
        collection = json.loads(
            (staging / "collection_manifest.json").read_text()
        )
        exposure_payload = json.loads(
            (staging / "mayo_exposure_manifest.json").read_text()
        )
        expected_counts = dict(collection["counts"])
        expected_collection_classification = collection[
            "classification_integrity_id"
        ]
        expected_exposure_classification = exposure_payload[
            "classification_integrity_id"
        ]
        output = root / "cache"
        exposure = root / "mayo_exposure_manifest.json"

        def interrupt(phase):
            if phase == "committed":
                raise SimulatedProcessDeath(phase)

        try:
            with builder.output_parent_lock(output):
                builder.promote_generation(
                    staging, output, exposure_manifest_path=exposure,
                    phase_hook=interrupt, salt=salt,
                    expected_inventory_counts=expected_counts,
                    expected_collection_classification_integrity_id=(
                        expected_collection_classification
                    ),
                    expected_classification_integrity_id=(
                        expected_exposure_classification
                    ),
                )
        except SimulatedProcessDeath:
            pass
        else:
            raise AssertionError("committed interruption did not occur")

        _swap_equal_total_legacy_classes(output, salt)
        forged_commitment = builder._validate_staging(output, salt=salt)
        journal_path = root / ".cache.transaction.json"
        journal = json.loads(journal_path.read_text())
        journal["generation_commitment"] = forged_commitment
        journal_path.write_text(json.dumps(journal, sort_keys=True))
        journal_path.chmod(0o600)
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure, salt=salt,
                    expected_inventory_counts=expected_counts,
                    expected_collection_classification_integrity_id=(
                        expected_collection_classification
                    ),
                    expected_classification_integrity_id=(
                        expected_exposure_classification
                    ),
                ),
                ValueError,
                "forged committed journal cannot rewrite legacy classification",
            )


def test_sparse_int64_timelines_never_construct_source_span_ranges(c: Check):
    huge = np.iinfo(np.int64).max - 1
    indices = np.asarray([0, huge], dtype=np.int64)
    media = builder.MayoMediaSequence(
        features=np.zeros((2, 95), dtype=np.float32),
        valid_mask=np.ones(2, dtype=bool),
        timestamps=indices.astype(np.float64) / 60.0,
        source_frame_indices=indices,
        facial_transforms=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        facial_transform_mask=np.ones(2, dtype=bool),
        transform_source="same_detection_mediapipe_video_mode",
    )
    arkit = builder.ARKitSequence(
        features=np.zeros((2, 52), dtype=np.float32),
        valid_mask=np.ones(2, dtype=bool),
        timestamps=indices.astype(np.float64) / 60.0,
        source_frame_indices=indices,
    )
    real_range = range

    class SuperlinearRange(RuntimeError):
        pass

    def guarded_range(*args):
        stop = args[0] if len(args) == 1 else args[1]
        if stop > 32:
            raise SuperlinearRange("source-index span was used as an iteration bound")
        return real_range(*args)

    builder.range = guarded_range
    try:
        for operation in (
            lambda: builder.downsample_to_30hz(media),
            lambda: builder.downsample_arkit_to_30hz(arkit),
        ):
            try:
                view = operation()
            except ValueError:
                pass
            else:
                c.eq(view.source_frame_indices.tolist(), [0, huge],
                     "exact integer selection preserves an extreme observed row")
                c.eq(view.target_frame_indices.tolist(), [0, huge // 2],
                     "target index is computed without float precision loss")
    finally:
        del builder.range


def test_transaction_journal_recovers_simulated_process_interruptions(c: Check):
    class SimulatedProcessDeath(BaseException):
        pass

    phases = (
        "old_output_moved",
        "old_exposure_moved",
        "new_output_installed",
        "new_exposure_installed",
    )
    with tempfile.TemporaryDirectory() as td:
        outer = Path(td)
        for phase_to_interrupt in phases:
            root = outer / phase_to_interrupt
            root.mkdir()
            root.chmod(0o700)
            output = root / "cache"
            output.mkdir()
            output.chmod(0o700)
            (output / "sentinel").write_text("old-cache")
            (output / "sentinel").chmod(0o600)
            exposure = root / "mayo_exposure_manifest.json"
            exposure.write_text("old-exposure")
            exposure.chmod(0o600)
            staging = _canonical_transaction_staging(
                root, ".cache.staging-new",
                b"interrupt-recovery-salt-0123456789",
            )

            def interrupt(phase):
                if phase == phase_to_interrupt:
                    raise SimulatedProcessDeath(phase)

            try:
                with builder.output_parent_lock(output):
                    builder.promote_generation(
                        staging, output, exposure_manifest_path=exposure,
                        phase_hook=interrupt,
                    )
            except SimulatedProcessDeath:
                pass
            else:
                raise AssertionError("simulated process death did not interrupt promotion")

            journal = root / ".cache.transaction.json"
            c.true(journal.is_file(), f"{phase_to_interrupt} leaves a recovery journal")
            with builder.output_parent_lock(output):
                builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure
                )
            c.eq((output / "sentinel").read_text(), "old-cache",
                 f"{phase_to_interrupt} restores the old complete cache")
            c.eq(exposure.read_text(), "old-exposure",
                 f"{phase_to_interrupt} restores the matching exposure ledger")
            c.true(not journal.exists(), "successful recovery removes the journal")
            c.true(not any(root.glob(".cache.*-*")),
                   "successful recovery removes transaction staging/backups")


def test_committed_key_drift_downgrade_write_failure_preserves_indeterminate_evidence(
    c: Check,
):
    original_salt = b"committed-key-drift-original-012"
    replacement_salt = b"committed-key-drift-replace-0123"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)
        output = root / "cache"
        output.mkdir(mode=0o700)
        (output / "old-generation-sentinel").write_text("old-cache")
        (output / "old-generation-sentinel").chmod(0o600)
        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)
        staging = _canonical_transaction_staging(
            root, ".cache.staging-key-drift", original_salt,
        )
        key = (
            root / "outputs" / "dynamic_landmark" / "pretraining"
            / ".mayo_ssl_hmac.key"
        )
        key.parent.mkdir(parents=True)
        (root / "outputs" / "dynamic_landmark").chmod(0o700)
        key.parent.chmod(0o700)
        key.write_bytes(original_salt)
        key.chmod(0o600)

        real_write_journal = builder._write_transaction_journal
        committed_was_durable = False

        def fail_only_downgrade(path, payload):
            nonlocal committed_was_durable
            if (
                committed_was_durable
                and payload["phase"] == "new_exposure_installed"
            ):
                raise OSError("forced journal downgrade write failure")
            real_write_journal(path, payload)
            if payload["phase"] == "committed":
                committed_was_durable = True

        def drift_key_after_committed(phase):
            if phase != "committed":
                return
            replacement = key.parent / ".replacement-mayo-key"
            replacement.write_bytes(replacement_salt)
            replacement.chmod(0o600)
            os.replace(replacement, key)

        caught: BaseException | None = None
        builder._write_transaction_journal = fail_only_downgrade
        try:
            try:
                with builder._hold_canonical_mayo_key(
                    key, project_root=root,
                ) as held_key, builder.output_parent_lock(output):
                    builder.promote_generation(
                        staging,
                        output,
                        exposure_manifest_path=exposure,
                        salt=held_key.key_bytes,
                        phase_hook=drift_key_after_committed,
                        continuity_validator=held_key.assert_unchanged,
                    )
            except BaseException as exc:  # inspect the complete chained failure
                caught = exc
        finally:
            builder._write_transaction_journal = real_write_journal

        journal_path = root / ".cache.transaction.json"
        c.true(
            journal_path.is_file(),
            "a failed committed-journal downgrade preserves the durable journal",
        )
        journal = json.loads(journal_path.read_text())
        token = journal["token"]
        output_backup = root / f".cache.backup-{token}"
        exposure_backup = root / f".mayo_exposure_manifest.json.backup-{token}"
        c.eq(journal["phase"], "committed",
             "the last durable phase remains explicit and indeterminate")
        c.true(
            output.is_dir()
            and not (output / "old-generation-sentinel").exists()
            and (output / "collection_manifest.json").is_file(),
            "the newly installed canonical generation remains as evidence",
        )
        c.eq(
            (output_backup / "old-generation-sentinel").read_text(),
            "old-cache",
            "the previous cache generation backup remains recoverable",
        )
        c.eq(
            exposure_backup.read_text(),
            "old-exposure",
            "the previous exposure backup remains recoverable",
        )
        c.true(
            exposure.read_text() != "old-exposure",
            "the newly installed canonical exposure remains as evidence",
        )
        c.true(
            isinstance(caught, ValueError)
            and _exception_chain_contains(
                caught, ValueError, "HMAC salt must have exactly one hard link"
            ),
            "key drift remains the primary publication failure",
        )
        c.true(
            caught is not None and _exception_chain_contains(
                caught, OSError, "forced journal downgrade write failure"
            ),
            "the journal downgrade storage failure remains in the exception chain",
        )

        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output,
                    exposure_manifest_path=exposure,
                    salt=replacement_salt,
                ),
                ValueError,
                "retry with the drifted key fails closed",
            )
        c.true(
            journal_path.is_file()
            and output_backup.is_dir()
            and exposure_backup.is_file(),
            "a failed retry preserves all indeterminate transaction evidence",
        )


def test_committed_key_drift_downgrade_fsync_ambiguity_blocks_recovery(c: Check):
    original_salt = b"committed-key-drift-original-012"
    replacement_salt = b"committed-key-drift-replace-0123"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)
        output = root / "cache"
        output.mkdir(mode=0o700)
        sentinel = output / "old-generation-sentinel"
        sentinel.write_text("old-cache")
        sentinel.chmod(0o600)
        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)
        staging = _canonical_transaction_staging(
            root, ".cache.staging-key-drift-fsync", original_salt,
        )
        key = (
            root / "outputs" / "dynamic_landmark" / "pretraining"
            / ".mayo_ssl_hmac.key"
        )
        key.parent.mkdir(parents=True)
        (root / "outputs" / "dynamic_landmark").chmod(0o700)
        key.parent.chmod(0o700)
        key.write_bytes(original_salt)
        key.chmod(0o600)
        journal_path = root / ".cache.transaction.json"
        original_fsync_directory = builder._fsync_directory
        committed_seen = False
        fsync_faulted = False

        def drift_key_after_committed(phase):
            nonlocal committed_seen
            if phase != "committed":
                return
            committed_seen = True
            replacement = key.parent / ".replacement-mayo-key"
            replacement.write_bytes(replacement_salt)
            replacement.chmod(0o600)
            os.replace(replacement, key)

        def fail_after_downgrade_replace(path):
            nonlocal fsync_faulted
            if committed_seen and Path(path) == root and journal_path.is_file():
                live = json.loads(journal_path.read_text())
                if live.get("phase") == "new_exposure_installed":
                    fsync_faulted = True
                    raise OSError("forced post-replace journal fsync failure")
            return original_fsync_directory(path)

        caught: BaseException | None = None
        builder._fsync_directory = fail_after_downgrade_replace
        try:
            try:
                with builder._hold_canonical_mayo_key(
                    key, project_root=root,
                ) as held_key, builder.output_parent_lock(output):
                    builder.promote_generation(
                        staging,
                        output,
                        exposure_manifest_path=exposure,
                        salt=held_key.key_bytes,
                        phase_hook=drift_key_after_committed,
                        continuity_validator=held_key.assert_unchanged,
                    )
            except BaseException as exc:
                caught = exc
        finally:
            builder._fsync_directory = original_fsync_directory

        c.true(fsync_faulted, "fault occurs after downgrade rename")
        c.true(
            caught is not None and _exception_chain_contains(
                caught, OSError, "forced post-replace journal fsync failure"
            ),
            "downgrade fsync ambiguity remains in the exception chain",
        )
        live = json.loads(journal_path.read_text())
        token = live["token"]
        output_backup = root / f".cache.backup-{token}"
        exposure_backup = root / f".mayo_exposure_manifest.json.backup-{token}"
        c.true(live.get("indeterminate") is True,
               "ambiguous downgrade is durably marked indeterminate")
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output,
                    exposure_manifest_path=exposure,
                    salt=replacement_salt,
                ),
                RuntimeError,
                "automatic recovery refuses ambiguous downgrade evidence",
            )
        c.true(
            journal_path.is_file()
            and output.is_dir()
            and exposure.is_file()
            and output_backup.is_dir()
            and exposure_backup.is_file(),
            "retry preserves both canonical and prior-generation evidence",
        )


def test_coupled_promotion_rejects_unsafe_existing_storage(c: Check):
    for scenario in (
        "output-mode", "nested-file-mode", "nested-file-hardlink",
        "exposure-mode", "exposure-hardlink",
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.chmod(0o700)
            output = root / "cache"
            output.mkdir(mode=0o700)
            sentinel = output / "old-generation-sentinel"
            sentinel.write_text("old-cache")
            sentinel.chmod(0o600)
            exposure = root / "mayo_exposure_manifest.json"
            exposure.write_text("old-exposure")
            exposure.chmod(0o600)
            staging = _canonical_transaction_staging(
                root, ".cache.staging-unsafe-existing",
                b"unsafe-existing-storage-salt-012",
            )
            alias: Path | None = None
            if scenario == "output-mode":
                output.chmod(0o777)
            elif scenario == "nested-file-mode":
                sentinel.chmod(0o666)
            elif scenario == "nested-file-hardlink":
                alias = root / ".old-cache-hardlink"
                os.link(sentinel, alias)
            elif scenario == "exposure-mode":
                exposure.chmod(0o666)
            else:
                alias = root / ".exposure-hardlink"
                os.link(exposure, alias)
            with builder.output_parent_lock(output):
                c.raises(
                    lambda: builder.promote_generation(
                        staging, output, exposure_manifest_path=exposure,
                    ),
                    ValueError,
                    f"{scenario} is rejected rather than replaced",
                )
            c.true(staging.is_dir(), f"{scenario} leaves staging untouched")
            c.eq(sentinel.read_text(), "old-cache")
            c.eq(exposure.read_text(), "old-exposure")
            c.true(not (root / ".cache.transaction.json").exists())
            if alias is not None:
                alias.unlink()


def test_coupled_promotion_rechecks_old_tree_after_move(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)
        output = root / "cache"
        output.mkdir(mode=0o700)
        sentinel = output / "old-generation-sentinel"
        sentinel.write_text("old-cache")
        sentinel.chmod(0o600)
        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)
        staging = _canonical_transaction_staging(
            root, ".cache.staging-old-tree-race",
            b"old-tree-race-storage-salt-01234",
        )

        def mutate_during_old_output_move(source, destination):
            if Path(source) == output and ".backup-" in Path(destination).name:
                sentinel.chmod(0o666)
            return os.replace(source, destination)

        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.promote_generation(
                    staging,
                    output,
                    exposure_manifest_path=exposure,
                    replace_func=mutate_during_old_output_move,
                ),
                ValueError,
                "old generation mode drift inside rename hook fails closed",
            )
        journal = root / ".cache.transaction.json"
        c.true(journal.is_file(), "move drift retains a transaction journal")
        retained = json.loads(journal.read_text())
        c.true(retained["indeterminate"] is True)
        backup = root / f".cache.backup-{retained['token']}"
        c.true(not output.exists(), "polluted generation is not restored canonical")
        c.eq((backup / sentinel.name).read_text(), "old-cache")
        c.eq(stat.S_IMODE((backup / sentinel.name).stat().st_mode), 0o666)
        c.eq(exposure.read_text(), "old-exposure")
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure,
                ),
                RuntimeError,
                "automatic recovery refuses polluted retained evidence",
            )
        c.true(journal.is_file() and backup.is_dir())


def test_coupled_promotion_rechecks_old_exposure_after_move(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)
        output = root / "cache"
        output.mkdir(mode=0o700)
        sentinel = output / "old-generation-sentinel"
        sentinel.write_text("old-cache")
        sentinel.chmod(0o600)
        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)
        staging = _canonical_transaction_staging(
            root, ".cache.staging-old-exposure-race",
            b"old-exposure-race-storage-salt-0123",
        )

        def mutate_during_old_exposure_move(source, destination):
            if Path(source) == exposure and ".backup-" in Path(destination).name:
                exposure.chmod(0o666)
            return os.replace(source, destination)

        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.promote_generation(
                    staging,
                    output,
                    exposure_manifest_path=exposure,
                    replace_func=mutate_during_old_exposure_move,
                ),
                ValueError,
                "old exposure mode drift inside rename hook fails closed",
            )
        journal = root / ".cache.transaction.json"
        c.true(journal.is_file(), "move drift retains a transaction journal")
        retained = json.loads(journal.read_text())
        c.true(retained["indeterminate"] is True)
        output_backup = root / f".cache.backup-{retained['token']}"
        exposure_backup = root / (
            f".mayo_exposure_manifest.json.backup-{retained['token']}"
        )
        c.true(not output.exists() and not exposure.exists())
        c.eq((output_backup / sentinel.name).read_text(), "old-cache")
        c.eq(exposure_backup.read_text(), "old-exposure")
        c.eq(stat.S_IMODE(exposure_backup.stat().st_mode), 0o666)
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure,
                ),
                RuntimeError,
                "automatic recovery refuses polluted retained evidence",
            )
        c.true(
            journal.is_file()
            and output_backup.is_dir()
            and exposure_backup.is_file()
        )


def test_coupled_promotion_retains_hardlinked_old_tree_evidence(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)
        output = root / "cache"
        output.mkdir(mode=0o700)
        sentinel = output / "old-generation-sentinel"
        sentinel.write_text("old-cache")
        sentinel.chmod(0o600)
        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)
        staging = _canonical_transaction_staging(
            root, ".cache.staging-old-tree-hardlink-race",
            b"old-tree-hardlink-race-salt-01234",
        )
        alias = root / ".attacker-hardlink"

        def hardlink_during_old_output_move(source, destination):
            if Path(source) == output and ".backup-" in Path(destination).name:
                os.link(sentinel, alias)
            return os.replace(source, destination)

        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.promote_generation(
                    staging,
                    output,
                    exposure_manifest_path=exposure,
                    replace_func=hardlink_during_old_output_move,
                ),
                ValueError,
                "old generation hardlink drift fails closed",
            )
        journal = root / ".cache.transaction.json"
        c.true(journal.is_file(), "hardlink drift retains transaction evidence")
        retained = json.loads(journal.read_text())
        c.true(retained["indeterminate"] is True)
        backup = root / f".cache.backup-{retained['token']}"
        retained_sentinel = backup / sentinel.name
        c.true(not output.exists() and backup.is_dir() and alias.is_file())
        c.eq(retained_sentinel.stat().st_nlink, 2)
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure,
                ),
                RuntimeError,
                "automatic recovery refuses hardlinked retained evidence",
            )
        c.true(journal.is_file() and backup.is_dir() and alias.is_file())


def test_recovery_rejects_old_backup_polluted_after_process_interruption(c: Check):
    class SimulatedProcessDeath(BaseException):
        pass

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)
        output = root / "cache"
        output.mkdir(mode=0o700)
        sentinel = output / "old-generation-sentinel"
        sentinel.write_text("old-cache")
        sentinel.chmod(0o600)
        exposure = root / "mayo_exposure_manifest.json"
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)
        staging = _canonical_transaction_staging(
            root, ".cache.staging-interrupted-backup-race",
            b"interrupted-backup-race-salt-01234",
        )

        def interrupt_after_old_output_move(phase):
            if phase == "old_output_moved":
                raise SimulatedProcessDeath(phase)

        try:
            with builder.output_parent_lock(output):
                builder.promote_generation(
                    staging,
                    output,
                    exposure_manifest_path=exposure,
                    phase_hook=interrupt_after_old_output_move,
                )
        except SimulatedProcessDeath:
            pass
        else:
            raise AssertionError("promotion did not stop after the old output move")

        journal = root / ".cache.transaction.json"
        retained = json.loads(journal.read_text())
        c.true(retained["indeterminate"] is False)
        backup = root / f".cache.backup-{retained['token']}"
        retained_sentinel = backup / sentinel.name
        retained_sentinel.chmod(0o666)
        with builder.output_parent_lock(output):
            c.raises(
                lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure,
                ),
                ValueError,
                "recovery preflights backup privacy before any restoration",
            )
        c.true(not output.exists(), "polluted backup is not restored canonical")
        c.eq(stat.S_IMODE(retained_sentinel.stat().st_mode), 0o666)
        c.true(journal.is_file() and backup.is_dir())
        c.eq(exposure.read_text(), "old-exposure")


def test_committed_recovery_revalidates_all_generation_commitments(c: Check):
    class SimulatedProcessDeath(BaseException):
        pass

    salt = b"transaction-integrity-salt-0123456789"
    scenarios = ("cache_deleted", "cache_changed", "manifest_changed", "exposure_changed")
    with tempfile.TemporaryDirectory() as td:
        outer = Path(td)
        for scenario in scenarios:
            root = outer / scenario
            root.mkdir()
            root.chmod(0o700)
            output = root / "cache"
            exposure = root / "mayo_exposure_manifest.json"
            staging = _canonical_transaction_staging(
                root, ".cache.staging-committed", salt
            )

            def interrupt(phase):
                if phase == "committed":
                    raise SimulatedProcessDeath(phase)

            try:
                with builder.output_parent_lock(output):
                    builder.promote_generation(
                        staging, output, exposure_manifest_path=exposure,
                        phase_hook=interrupt,
                    )
            except SimulatedProcessDeath:
                pass
            else:
                raise AssertionError("committed phase was not interrupted")

            cache_path = next((output / "mediapipe").glob("*.npz"))
            if scenario == "cache_deleted":
                cache_path.unlink()
            elif scenario == "cache_changed":
                cache_path.write_bytes(b"tampered-cache")
            elif scenario == "manifest_changed":
                (output / "collection_manifest.json").write_text("{}")
            else:
                exposure.write_text("{}")
            journal = root / ".cache.transaction.json"
            with builder.output_parent_lock(output):
                c.raises(lambda: builder.recover_interrupted_generations(
                    output, exposure_manifest_path=exposure
                ), ValueError,
                         f"{scenario} makes committed recovery fail closed")
            c.true(journal.is_file(),
                   f"{scenario} preserves the private journal for investigation")


def test_generation_tree_is_directory_fsynced_before_promotion(c: Check):
    class SimulatedProcessDeath(BaseException):
        pass

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        output = root / "cache"
        exposure = root / "mayo_exposure_manifest.json"
        staging = _canonical_transaction_staging(
            root, ".cache.staging-fsync", b"directory-fsync-salt-0123456789012"
        )
        real_open = builder.os.open
        real_fsync = builder.os.fsync
        descriptor_paths = {}
        events = []

        def monitored_open(path, flags, *args, **kwargs):
            descriptor = real_open(path, flags, *args, **kwargs)
            if flags & getattr(os, "O_DIRECTORY", 0):
                descriptor_paths[descriptor] = Path(path)
                c.true(bool(flags & getattr(os, "O_NOFOLLOW", 0)),
                       "generation directories are opened no-follow before fsync")
            return descriptor

        def monitored_fsync(descriptor):
            path = descriptor_paths.get(descriptor)
            if path is not None:
                events.append(path)
            return real_fsync(descriptor)

        def interrupt(phase):
            if phase == "prepared":
                events.append("prepared")
                raise SimulatedProcessDeath(phase)

        builder.os.open = monitored_open
        builder.os.fsync = monitored_fsync
        try:
            try:
                with builder.output_parent_lock(output):
                    builder.promote_generation(
                        staging, output, exposure_manifest_path=exposure,
                        phase_hook=interrupt,
                    )
            except SimulatedProcessDeath:
                pass
            else:
                raise AssertionError("prepared phase was not interrupted")
        finally:
            builder.os.open = real_open
            builder.os.fsync = real_fsync
        prepared_index = events.index("prepared")
        c.true(all(path in events[:prepared_index] for path in (
            staging / "mediapipe", staging / "arkit", staging,
        )), "both cache subdirectories and staging root are durable before promotion")


def test_run_builder_uses_pinned_homogeneous_sources_and_exact_runtime(c: Check):
    class OneFrameCapture:
        def __init__(self):
            self.read_count = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 59.95,
                cv2.CAP_PROP_FRAME_COUNT: 1.0,
                cv2.CAP_PROP_FRAME_WIDTH: 5.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 4.0,
            }.get(prop, 0.0)

        def read(self):
            if self.read_count:
                return False, None
            self.read_count += 1
            return True, np.zeros((4, 5, 3), np.uint8)

        def release(self):
            self.released = True

    class FakeVideoExtractor:
        feature_schema = DYNAMIC_FEATURE_SCHEMA
        feature_names = list(DYNAMIC_FEATURE_NAMES)

        def __init__(self, model_path):
            self.model_path = Path(model_path)
            self.closed = 0

        def extract_video_frame(self, _frame, _timestamp_ms):
            return (np.ones(95, dtype=np.float32), None,
                    np.eye(4, dtype=np.float32))

        def close(self):
            self.closed += 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / "data" / "livelinkface_data"
        exports = root / "data" / "mediapipe_out"
        session = data / "PHI_session"
        legacy = exports / "PHI_session"
        session.mkdir(parents=True)
        legacy.mkdir(parents=True)
        source = session / "private.mov"
        source.write_bytes(b"one-frame-video")
        asset = builder.VideoAsset(
            session, source, builder.VideoMetadata(1, 59.95, 5, 4),
            _sha(b"one-frame-video"), legacy,
        )
        counts = {
            "total_sessions": 1, "video_bearing_sessions": 1,
            "without_video_sessions": 0, "exact_duplicate_copies_excluded": 0,
            "short_qc_clips_excluded": 0, "long_unique_videos": 1,
            "existing_complete_v2_exports": 1, "remaining_long_videos": 0,
            "remaining_long_video_frames": 0, "arkit_only_sessions": 0,
            "arkit_trajectories": 0, "arkit_rows": 0,
            "arkit_timecode_gaps": 0, "metadata_only_sessions": 0,
        }
        inventory = builder.MayoInventory(
            data, exports, counts, (asset,), (asset,), (asset,), (), (), (), (), (), (),
        )

        model = root / "face_landmarker.task"
        model.write_bytes(b"model")
        expected_python = root / "isolated" / "bin" / "python"
        expected_python.parent.mkdir(parents=True)
        expected_python.write_bytes(b"python-runtime")
        versions = {
            "numpy": "1.2.3", "mediapipe": "0.10.35", "opencv-python": "4.9.0",
        }

        def version_resolver(name):
            if name not in versions:
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        artifacts = {}
        fake_site_packages = (
            expected_python.parent.parent / "lib" / "python3.10" / "site-packages"
        )
        fake_site_packages.mkdir(parents=True)
        for distribution, version in versions.items():
            dist_info = fake_site_packages / f"{distribution}.dist-info"
            dist_info.mkdir()
            metadata = dist_info / "METADATA"
            record = dist_info / "RECORD"
            installed = fake_site_packages / distribution
            installed_bytes = f"installed-{distribution}".encode()
            installed.write_bytes(installed_bytes)
            metadata_bytes = f"Name: {distribution}\nVersion: {version}\n".encode()
            metadata.write_bytes(metadata_bytes)
            with record.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows((
                    (distribution, _record_digest(installed_bytes),
                     str(len(installed_bytes))),
                    (f"{distribution}.dist-info/METADATA",
                     _record_digest(metadata_bytes), str(len(metadata_bytes))),
                    (f"{distribution}.dist-info/RECORD", "", ""),
                ))
            artifacts[distribution] = (metadata, record)

        output = root / "outputs" / "dynamic_landmark" / "pretraining" / "mayo_ssl_cache"
        exposure = root / "outputs" / "dynamic_landmark" / "mayo_exposure_manifest.json"
        salt = output.parent / ".mayo_ssl_hmac.key"
        salt.parent.mkdir(parents=True)
        (root / "outputs" / "dynamic_landmark").chmod(0o700)
        output.parent.chmod(0o700)
        salt.write_bytes(b"k" * 32)
        salt.chmod(0o600)
        captures = []
        extractors = []

        def capture_factory(path):
            pinned = Path(path)
            c.true(".source_snapshots" in pinned.parts)
            c.eq(os.stat(pinned).st_ino, os.stat(source).st_ino,
                 "the decoder opens the hard-linked audited inode")
            capture = OneFrameCapture()
            captures.append(capture)
            return capture

        def extractor_factory(**kwargs):
            pinned_model = Path(kwargs["model_path"])
            c.true(".source_snapshots" in pinned_model.parts)
            item = FakeVideoExtractor(pinned_model)
            extractors.append(item)
            return item

        previous_umask = os.umask(0o777)
        try:
            manifest = builder._run_builder_impl(
                data, exports, model, salt, output, exposure,
                extractor_factory=extractor_factory,
                capture_factory=capture_factory,
                inventory_factory=lambda *_args, **_kwargs: inventory,
                project_root=root,
                current_executable=expected_python,
                expected_executable=expected_python,
                version_resolver=version_resolver,
                dependency_artifact_resolver=lambda name: artifacts[name],
                provenance_python_executable=expected_python,
            )
        finally:
            os.umask(previous_umask)
        c.eq(manifest["mediapipe_records"][0]["legacy_export_audit_status"],
             "not_reused_unverifiable_source_binding")
        c.eq(len(captures), 1, "legacy export was not opened or compacted")
        c.true(captures[0].released and extractors[0].closed == 1)
        c.true(output.is_dir() and exposure.is_file())
        private_directories = (
            root / "outputs" / "dynamic_landmark",
            output.parent,
            output,
            output / "mediapipe",
            output / "arkit",
        )
        c.true(all(
            stat.S_IMODE(path.stat().st_mode) == 0o700
            and path.stat().st_uid == os.geteuid()
            for path in private_directories
        ), "hostile umask cannot weaken any committed private directory")
        private_files = (
            salt,
            exposure,
            output.parent / f".{output.name}.lock",
            *(path for path in output.rglob("*") if path.is_file()),
        )
        c.true(all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            and path.stat().st_uid == os.geteuid()
            and path.stat().st_nlink == 1
            for path in private_files
        ), "hostile umask cannot weaken any committed private file")
        c.true(not any(output.rglob(".source_snapshots")),
               "pinned raw hard links never enter the promoted cache")
        saved_manifest = json.loads((output / "collection_manifest.json").read_text())
        c.true("dependency_sha256" not in saved_manifest["provenance"],
               "public provenance does not expose per-file dependency digests")
        c.true("dependency_aggregate_sha256" in saved_manifest["provenance"])
        dependency_rows = saved_manifest["provenance"]["runtime_dependencies"]
        c.eq({row["distribution"]: row["installed_file_count"]
              for row in dependency_rows}, {
                  "python": 1, "numpy": 3, "mediapipe": 3, "opencv-python": 3,
              })
        public_text = (
            (output / "collection_manifest.json").read_text()
            + exposure.read_text()
        )
        raw_cache_digest = _sha(next((output / "mediapipe").glob("*.npz")).read_bytes())
        c.true(asset.source_sha256 not in public_text
               and raw_cache_digest not in public_text
               and salt.read_bytes().hex() not in public_text,
               "ignored generation JSON contains no raw source/cache digest or salt")


def _assert_key_fault_blocks_publication(
    c: Check,
    root: Path,
    fault: str,
    injection_point: str = "extraction",
    *,
    with_existing_generation: bool = False,
) -> None:
    class OneFrameCapture:
        def __init__(self):
            self.read_count = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 60.0,
                cv2.CAP_PROP_FRAME_COUNT: 1.0,
                cv2.CAP_PROP_FRAME_WIDTH: 5.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 4.0,
            }.get(prop, 0.0)

        def read(self):
            if self.read_count:
                return False, None
            self.read_count += 1
            return True, np.zeros((4, 5, 3), np.uint8)

        def release(self):
            return None

    data = root / "data" / "livelinkface_data"
    exports = root / "data" / "mediapipe_out"
    session = data / "PHI_session"
    session.mkdir(parents=True)
    exports.mkdir(parents=True)
    source = session / "private.mov"
    source.write_bytes(b"one-frame-video")
    asset = builder.VideoAsset(
        session,
        source,
        builder.VideoMetadata(1, 60.0, 5, 4),
        _sha(source.read_bytes()),
        None,
    )
    counts = {
        "total_sessions": 1, "video_bearing_sessions": 1,
        "without_video_sessions": 0, "exact_duplicate_copies_excluded": 0,
        "short_qc_clips_excluded": 0, "long_unique_videos": 1,
        "existing_complete_v2_exports": 0, "remaining_long_videos": 1,
        "remaining_long_video_frames": 1, "arkit_only_sessions": 0,
        "arkit_trajectories": 0, "arkit_rows": 0,
        "arkit_timecode_gaps": 0, "metadata_only_sessions": 0,
    }
    inventory = builder.MayoInventory(
        data, exports, counts, (asset,), (asset,), (), (asset,), (), (), (), (), (),
    )
    model = root / "face_landmarker.task"
    model.write_bytes(b"model")
    expected_python = root / "isolated" / "bin" / "python"
    expected_python.parent.mkdir(parents=True)
    expected_python.write_bytes(b"python-runtime")
    output = (
        root / "outputs" / "dynamic_landmark" / "pretraining"
        / "mayo_ssl_cache"
    )
    exposure = (
        root / "outputs" / "dynamic_landmark"
        / "mayo_exposure_manifest.json"
    )
    key = output.parent / ".mayo_ssl_hmac.key"
    key.parent.mkdir(parents=True)
    (root / "outputs" / "dynamic_landmark").chmod(0o700)
    key.parent.chmod(0o700)
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    if with_existing_generation:
        output.mkdir(mode=0o700)
        (output / "old-generation-sentinel").write_text("old-cache")
        (output / "old-generation-sentinel").chmod(0o600)
        exposure.write_text("old-exposure")
        exposure.chmod(0o600)

    dependency_specs = (
        ("mediapipe", "mediapipe", "mediapipe==0.10.35"),
        ("numpy", "numpy", "numpy==1.26.4"),
        ("opencv", "opencv-python", "opencv-python==4.11.0"),
        ("python", "python", "python==3.10.2"),
    )
    dependency_files = tuple(
        builder.DependencyFileSnapshot(
            logical_name=logical_name,
            distribution=distribution,
            record_name=f"{distribution}/fixture.py",
            path=model,
            sha256=str(index + 1) * 64,
            device=1,
            inode=index + 1,
            size=1,
            mtime_ns=1,
        )
        for index, (logical_name, distribution, _requirement)
        in enumerate(dependency_specs)
    )
    fake_provenance = builder.ProvenanceSnapshot(
        source_files=((source, asset.source_sha256),),
        model_file=(model, _sha(model.read_bytes())),
        producer_files=tuple(
            (name, SCRIPT, str(index + 5) * 64)
            for index, name in enumerate((
                "builder", "action_bundle", "clinical_landmarks",
                "dynamic_landmark_schema", "feature_registry",
            ))
        ),
        dependencies={
            logical_name: requirement
            for logical_name, _distribution, requirement in dependency_specs
        },
        dependency_distributions={
            logical_name: distribution
            for logical_name, distribution, _requirement in dependency_specs
        },
        dependency_files=dependency_files,
        dependency_aggregate_sha256="a" * 64,
        producer_aggregate_sha256="b" * 64,
        source_aggregate_sha256="c" * 64,
    )
    fault_injected = False

    def inject_fault() -> None:
        nonlocal fault_injected
        if fault_injected:
            return
        fault_injected = True
        if fault == "replacement":
            replacement = key.parent / ".replacement-mayo-key"
            replacement.write_bytes(b"k" * 32)
            replacement.chmod(0o600)
            os.replace(replacement, key)
        elif fault == "chmod":
            key.chmod(0o640)
        elif fault == "bytes":
            key.write_bytes(b"x" * 32)
            key.chmod(0o600)
        elif fault == "hardlink":
            os.link(key, key.parent / ".hardlinked-mayo-key")
        elif fault == "symlink":
            target = key.parent / ".symlink-target-mayo-key"
            target.write_bytes(b"k" * 32)
            target.chmod(0o600)
            key.unlink()
            key.symlink_to(target.name)
        elif fault == "short":
            key.write_bytes(b"s" * 31)
            key.chmod(0o600)
        elif fault == "long":
            key.write_bytes(b"l" * 33)
            key.chmod(0o600)
        else:
            raise AssertionError(f"unknown key fault {fault!r}")

    class FaultInjectingExtractor:
        feature_schema = DYNAMIC_FEATURE_SCHEMA
        feature_names = list(DYNAMIC_FEATURE_NAMES)

        def extract_video_frame(self, _frame, _timestamp_ms):
            if injection_point == "extraction":
                inject_fault()
            return (
                np.ones(95, dtype=np.float32),
                None,
                np.eye(4, dtype=np.float32),
            )

        def close(self):
            return None

    original_snapshot = builder.snapshot_provenance
    original_assert = builder.assert_provenance_unchanged
    original_promote = builder.promote_generation

    def promote_with_boundary_fault(*args, **kwargs):
        existing_hook = kwargs.get("phase_hook")

        def phase_hook(phase):
            if existing_hook is not None:
                existing_hook(phase)
            if phase == injection_point:
                inject_fault()

        kwargs["phase_hook"] = phase_hook
        return original_promote(*args, **kwargs)

    builder.snapshot_provenance = lambda *_args, **_kwargs: fake_provenance
    builder.assert_provenance_unchanged = lambda *_args, **_kwargs: None
    if injection_point != "extraction":
        builder.promote_generation = promote_with_boundary_fault
    rejected = False
    try:
        try:
            builder._run_builder_impl(
                data,
                exports,
                model,
                key,
                output,
                exposure,
                extractor_factory=lambda **_kwargs: FaultInjectingExtractor(),
                capture_factory=lambda _path: OneFrameCapture(),
                inventory_factory=lambda *_args, **_kwargs: inventory,
                project_root=root,
                current_executable=expected_python,
                expected_executable=expected_python,
                provenance_python_executable=expected_python,
            )
        except ValueError:
            rejected = True
    finally:
        builder.snapshot_provenance = original_snapshot
        builder.assert_provenance_unchanged = original_assert
        builder.promote_generation = original_promote
        if key.exists():
            key.chmod(0o600)
    c.true(
        fault_injected,
        f"{fault} fault occurs at {injection_point}",
    )
    c.true(rejected, f"{fault} of the canonical key fails the build closed")
    if with_existing_generation:
        c.eq(
            (output / "old-generation-sentinel").read_text(),
            "old-cache",
            f"{fault} restores the prior canonical Mayo cache",
        )
        c.eq(
            exposure.read_text(),
            "old-exposure",
            f"{fault} restores the prior canonical Mayo exposure",
        )
    else:
        c.true(
            not output.exists() and not exposure.exists(),
            f"{fault} cannot publish either canonical Mayo artifact",
        )
    c.true(
        not any(output.parent.glob(f".{output.name}.staging-*")),
        f"{fault} rejection cleans the uncommitted private staging generation",
    )


def test_extraction_time_mayo_key_replacement_blocks_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(
            c, Path(td), "replacement",
        )


def test_extraction_time_mayo_key_chmod_blocks_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(c, Path(td), "chmod")


def test_extraction_time_mayo_key_byte_mutation_blocks_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(c, Path(td), "bytes")


def test_new_exposure_install_key_mutation_rolls_back_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(
            c, Path(td), "bytes", "new_exposure_installed",
        )


def test_commit_boundary_mayo_key_replacement_rolls_back_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(
            c, Path(td), "replacement", "committed",
        )


def test_commit_boundary_mayo_key_chmod_rolls_back_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(
            c, Path(td), "chmod", "committed",
        )


def test_commit_boundary_mayo_key_byte_mutation_rolls_back_publication(c: Check):
    with tempfile.TemporaryDirectory() as td:
        _assert_key_fault_blocks_publication(
            c, Path(td), "bytes", "committed",
        )


def test_commit_boundary_key_storage_fault_matrix_restores_existing_generation(
    c: Check,
):
    for fault in ("hardlink", "symlink", "short", "long"):
        with tempfile.TemporaryDirectory() as td:
            _assert_key_fault_blocks_publication(
                c,
                Path(td),
                fault,
                "committed",
                with_existing_generation=True,
            )


def test_public_builder_cannot_override_frozen_security_dependencies(c: Check):
    parameters = tuple(inspect.signature(builder.run_builder).parameters)
    c.eq(parameters, (
        "data_root", "existing_export_root", "model_path", "salt_file",
        "output_root", "exposure_manifest",
    ), "public build API cannot override runtime, inventory, path policy, or provenance")


def test_output_locations_are_confined_to_ignored_mayo_paths(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        output = root / "outputs" / "dynamic_landmark" / "pretraining" / "mayo_ssl_cache"
        exposure = root / "outputs" / "dynamic_landmark" / "mayo_exposure_manifest.json"
        got_output, got_exposure = builder.validate_output_locations(
            output, exposure, project_root=root
        )
        c.eq(got_output, output)
        c.eq(got_exposure, exposure)
        c.raises(lambda: builder.validate_output_locations(
            root / "tracked" / "mayo_ssl_cache", exposure, project_root=root), ValueError,
                 "large biometric cache cannot be directed to a trackable location")
        c.raises(lambda: builder.validate_output_locations(
            output, root / "tracked" / "exposure.json", project_root=root), ValueError,
                 "the permanent exposure ledger must use its exact ignored path")
        suffix_copy = root / "other-repository" / "outputs" / "dynamic_landmark"
        c.raises(lambda: builder.validate_output_locations(
            suffix_copy / "pretraining" / "mayo_ssl_cache",
            suffix_copy / "mayo_exposure_manifest.json", project_root=root,
        ), ValueError, "matching suffixes outside PROJECT_ROOT are rejected")

        data = root / "data" / "livelinkface_data"
        exports = root / "data" / "mediapipe_out"
        data.mkdir(parents=True)
        exports.mkdir()
        builder.validate_source_output_separation(data, exports, output, exposure)
        c.raises(lambda: builder.validate_source_output_separation(
            root / "outputs", exports, output, exposure,
        ), ValueError, "derived cache cannot overlap the raw-data tree in either direction")


def test_canonical_salt_permissions_runtime_and_inventory_only_cli(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        salt = root / "outputs" / "dynamic_landmark" / "pretraining" / ".mayo_ssl_hmac.key"
        salt.parent.mkdir(parents=True)
        (root / "outputs" / "dynamic_landmark").chmod(0o700)
        salt.parent.chmod(0o700)
        salt.write_bytes(b"s" * 32)
        salt.chmod(0o600)
        real_os_open = builder.os.open
        observed_open_flags = []

        def monitored_open(path, flags, *args, **kwargs):
            observed_open_flags.append(flags)
            return real_os_open(path, flags, *args, **kwargs)

        builder.os.open = monitored_open
        try:
            c.eq(builder.read_canonical_salt(salt, project_root=root), b"s" * 32)
        finally:
            builder.os.open = real_os_open
        c.true(any(flags & getattr(os, "O_NOFOLLOW", 0)
                   for flags in observed_open_flags),
               "salt bytes are read from a no-follow file descriptor")
        salt.chmod(0o640)
        c.raises(lambda: builder.read_canonical_salt(salt, project_root=root), ValueError,
                 "salt must be owner-only mode 0600")
        salt.chmod(0o600)
        c.raises(lambda: builder.read_canonical_salt(
            salt, project_root=root, owner_uid=os.getuid() + 1,
        ), ValueError, "salt must belong to the current user")

        parser = builder._parser()
        args = parser.parse_args([
            "--data-root", str(root / "data"),
            "--existing-export-root", str(root / "exports"),
            "--inventory-only",
        ])
        c.true(args.inventory_only)
        c.true(args.model_path is None and args.salt_file is None
               and args.output_root is None and args.exposure_manifest is None,
               "inventory audit does not require extraction-only arguments")
        c.true(str(builder.PINNED_MEDIAPIPE_PYTHON) in parser.format_help(),
               "real CLI help pins the isolated MediaPipe runtime")

        expected_python = root / "isolated" / "bin" / "python"
        expected_python.parent.mkdir(parents=True)
        expected_python.write_bytes(b"python")
        builder.validate_extraction_runtime(
            expected_python, expected_executable=expected_python
        )
        other_python = root / "other-python"
        other_python.write_bytes(b"other")
        c.raises(lambda: builder.validate_extraction_runtime(
            other_python, expected_executable=expected_python,
        ), RuntimeError, "cache extraction fails outside the pinned isolated runtime")


def test_canonical_mayo_key_requires_exactly_32_bytes(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        key = (
            root / "outputs" / "dynamic_landmark" / "pretraining"
            / ".mayo_ssl_hmac.key"
        )
        key.parent.mkdir(parents=True)
        (root / "outputs" / "dynamic_landmark").chmod(0o700)
        key.parent.chmod(0o700)
        for length in (33, 31, 4096):
            key.write_bytes(b"k" * length)
            key.chmod(0o600)
            c.raises(
                lambda: builder.read_canonical_salt(key, project_root=root),
                ValueError,
                f"canonical Mayo key length {length} is rejected",
            )


def test_committed_mayo_generation_exposes_narrow_read_only_authorizer(c: Check):
    c.true(
        hasattr(builder, "authorize_committed_mayo_ssl_generation"),
        "the bridge requires a public live committed-generation authorizer",
    )


def test_mayo_manifests_reject_duplicate_json_keys_at_every_object_depth(c: Check):
    cases = (
        ("collection_manifest.json", "dataset"),
        ("collection_manifest.json", "recording_id"),
        ("mayo_exposure_manifest.json", "dataset"),
        ("mayo_exposure_manifest.json", "recording_id"),
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for index, (filename, field) in enumerate(cases):
            salt = b"d" * 32
            staging = _semantic_staging(
                root, f".mayo_ssl_cache.staging-duplicate-{index}", salt,
                include_arkit=True, include_exclusions=True,
            )
            target = staging / filename
            target.write_bytes(_prepend_duplicate_json_field(
                target.read_bytes(), field, "/private/PHI/hidden-source.mov"
            ))
            c.raises(
                lambda staging=staging, salt=salt: builder._validate_staging(
                    staging, salt=salt
                ),
                ValueError,
                f"{filename} rejects duplicate {field!r} even when last-value wins",
            )


def test_external_exposure_manifest_is_recursively_duplicate_key_safe(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        salt = b"e" * 32
        output = _semantic_staging(
            root, ".mayo_ssl_cache.staging-external-duplicate", salt,
            include_arkit=True, include_exclusions=True,
        )
        commitment = builder._validate_staging(output, salt=salt)
        external = root / "mayo_exposure_manifest.json"
        external_payload = _prepend_duplicate_json_field(
            (output / "mayo_exposure_manifest.json").read_bytes(),
            "recording_id",
            "/private/PHI/hidden-external-source.mov",
        )
        external.write_bytes(external_payload)
        commitment["exposure_manifest_sha256"] = _sha(external_payload)
        real_validate_staging = builder._validate_staging
        builder._validate_staging = lambda *_args, **_kwargs: dict(commitment)
        try:
            c.raises(
                lambda: builder._assert_committed_generation(
                    output, external, commitment, salt=salt
                ),
                ValueError,
                "the separately published exposure copy is decoded, not only hashed",
            )
        finally:
            builder._validate_staging = real_validate_staging


def test_mayo_transaction_journal_and_nested_commitment_reject_duplicate_keys(c: Check):
    commitment = {
        "schema": "mayo_cache_generation_commitment_v3",
        "collection_manifest_sha256": "1" * 64,
        "exposure_manifest_sha256": "2" * 64,
        "mediapipe_file_count": 1,
        "arkit_file_count": 1,
        "cache_file_count": 2,
        "cache_tree_aggregate_sha256": "3" * 64,
        "generation_aggregate_sha256": "4" * 64,
        "inventory_counts_sha256": "5" * 64,
        "collection_classification_integrity_id": "agg_" + "6" * 64,
        "exposure_classification_integrity_id": "agg_" + "7" * 64,
    }
    journal = {
        "schema": "mayo_cache_exposure_transaction_v2",
        "token": "0123456789abcdef",
        "staging_name": ".cache.staging-0123456789abcdef",
        "exposure_name": "mayo_exposure_manifest.json",
        "had_output": False,
        "had_exposure": False,
        "phase": "prepared",
        "indeterminate": False,
        "generation_commitment": commitment,
    }
    serialized = json.dumps(journal, sort_keys=True, separators=(",", ":"))
    payloads = (
        serialized.replace(
            '"staging_name":',
            '"staging_name":"/private/PHI/hidden-staging","staging_name":',
            1,
        ),
        serialized.replace(
            '"generation_commitment":{',
            '"generation_commitment":{"schema":"/private/PHI/hidden-commitment",',
            1,
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / ".cache.transaction.json"
        for payload in payloads:
            path.write_text(payload, encoding="utf-8")
            path.chmod(0o600)
            c.raises(
                lambda: builder._load_transaction_journal(path),
                ValueError,
                "journal and nested generation commitment reject duplicate keys",
            )


class _CommittedMayoAuthorizerFixture:
    def __init__(self, root: Path):
        self.root = root
        self.salt_bytes = b"m" * 32
        staging = _semantic_staging(
            root, ".mayo_ssl_cache.staging-fixture", self.salt_bytes,
            include_arkit=True, include_exclusions=True,
        )
        self.output = (
            root / "outputs" / "dynamic_landmark" / "pretraining"
            / "mayo_ssl_cache"
        )
        self.exposure = (
            root / "outputs" / "dynamic_landmark"
            / "mayo_exposure_manifest.json"
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.exposure.parent.mkdir(parents=True, exist_ok=True)
        self.exposure.parent.chmod(0o700)
        self.output.parent.chmod(0o700)
        os.replace(staging, self.output)
        self.exposure.write_bytes(
            (self.output / "mayo_exposure_manifest.json").read_bytes()
        )
        self.exposure.chmod(0o600)
        self.key = self.output.parent / ".mayo_ssl_hmac.key"
        self.key.write_bytes(self.salt_bytes)
        self.key.chmod(0o600)
        self.lock = self.output.parent / f".{self.output.name}.lock"
        self.lock.write_bytes(b"")
        self.lock.chmod(0o600)
        self.collection = json.loads(
            (self.output / "collection_manifest.json").read_text()
        )
        self.exposure_value = json.loads(self.exposure.read_text())
        self.counts = dict(self.collection["counts"])
        self.data = root / "data" / "livelinkface_data"
        self.exports = root / "data" / "mediapipe_out"
        self.data.mkdir(parents=True)
        self.exports.mkdir()
        self.inventory = builder.MayoInventory(
            self.data, self.exports, self.counts,
            (), (), (), (), (), (), (), (), (),
        )

    def __enter__(self):
        self.original_root = builder.PROJECT_ROOT
        self.original_frozen = builder.FROZEN_INVENTORY
        self.original_inventory = builder.inventory_mayo_sources
        self.original_manifest_builder = builder.build_public_manifests
        builder.PROJECT_ROOT = self.root
        builder.FROZEN_INVENTORY = self.counts
        builder.inventory_mayo_sources = lambda *_args, **_kwargs: self.inventory
        builder.build_public_manifests = lambda *_args, **_kwargs: (
            json.loads(json.dumps(self.collection)),
            json.loads(json.dumps(self.exposure_value)),
        )
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        builder.PROJECT_ROOT = self.original_root
        builder.FROZEN_INVENTORY = self.original_frozen
        builder.inventory_mayo_sources = self.original_inventory
        builder.build_public_manifests = self.original_manifest_builder

    def authorize(self):
        return builder.authorize_committed_mayo_ssl_generation(
            self.data, self.exports, self.key, self.output, self.exposure,
        )


def test_committed_mayo_authorizer_requires_exact_private_tree_modes(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            private_directories = (
                fixture.output.parent,
                fixture.output,
                fixture.output / "mediapipe",
                fixture.output / "arkit",
                fixture.exposure.parent,
            )
            private_files = (
                fixture.output / "collection_manifest.json",
                fixture.output / "mayo_exposure_manifest.json",
                fixture.exposure,
                *tuple((fixture.output / "mediapipe").glob("*.npz")),
                *tuple((fixture.output / "arkit").glob("*.npz")),
            )
            for directory in private_directories:
                directory.chmod(0o700)
            for path in private_files:
                path.chmod(0o600)
            c.eq(fixture.authorize().commitment["schema"],
                 "mayo_cache_generation_commitment_v3",
                 "exact owner-only Mayo cache tree remains authorizable")

            for directory in private_directories:
                directory.chmod(0o777)
                try:
                    c.raises(
                        fixture.authorize,
                        ValueError,
                        f"unsafe private directory mode is rejected: {directory.name}",
                    )
                finally:
                    directory.chmod(0o700)

            for path in private_files:
                path.chmod(0o666)
                try:
                    c.raises(
                        fixture.authorize,
                        ValueError,
                        f"unsafe private file mode is rejected: {path.name}",
                    )
                finally:
                    path.chmod(0o600)

            for index, path in enumerate(private_files):
                alias = root / f"private-cache-hardlink-{index}"
                os.link(path, alias)
                try:
                    c.raises(
                        fixture.authorize,
                        ValueError,
                        "committed Mayo private files reject additional hard links",
                    )
                finally:
                    alias.unlink()


def test_committed_mayo_authorizer_never_mixes_swapped_generation_roots(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            expected_commitment = builder._validate_staging(
                fixture.output, salt=fixture.salt_bytes,
            )
            expected_cache = next((fixture.output / "mediapipe").glob("*.npz"))
            expected_cache_sha256 = _sha(expected_cache.read_bytes())
            with np.load(expected_cache, allow_pickle=False) as cached:
                expected_feature = float(cached["features_30hz"][0, 0])

            alternate = _semantic_staging(
                root,
                ".mayo_ssl_cache.alternate-generation",
                fixture.salt_bytes,
                include_arkit=True,
                include_exclusions=True,
            )
            alternate_cache = next((alternate / "mediapipe").glob("*.npz"))

            def mutate(payload: dict[str, np.ndarray]) -> None:
                payload["features_source_rate"][0, 0] = np.float32(123.0)
                payload["features_30hz"][0, 0] = np.float32(123.0)

            _rewrite_npz(alternate_cache, mutate)
            _refresh_cache_integrity(
                alternate, fixture.salt_bytes, "mediapipe",
            )
            alternate_commitment = builder._validate_staging(
                alternate, salt=fixture.salt_bytes,
            )
            alternate_cache_sha256 = _sha(alternate_cache.read_bytes())
            c.true(
                alternate_commitment != expected_commitment
                and alternate_cache_sha256 != expected_cache_sha256,
                "the adversarial generation is independently valid but byte-distinct",
            )

            parked = fixture.output.parent / ".mayo_ssl_cache.parked-generation"
            original_validate = builder._validate_staging
            validation_calls = 0
            swapped = False

            def validate_with_root_swap(*args, **kwargs):
                nonlocal validation_calls, swapped
                validation_calls += 1
                if validation_calls == 2 and swapped:
                    os.rename(fixture.output, alternate)
                    os.rename(parked, fixture.output)
                    swapped = False
                result = original_validate(*args, **kwargs)
                if validation_calls == 1:
                    os.rename(fixture.output, parked)
                    os.rename(alternate, fixture.output)
                    swapped = True
                return result

            builder._validate_staging = validate_with_root_swap
            authorized = None
            rejected = False
            descriptor_baseline = (
                len(os.listdir("/dev/fd")) if Path("/dev/fd").is_dir() else None
            )
            try:
                try:
                    authorized = fixture.authorize()
                except (OSError, RuntimeError, ValueError):
                    rejected = True
            finally:
                builder._validate_staging = original_validate
                if swapped:
                    os.rename(fixture.output, alternate)
                    os.rename(parked, fixture.output)

            if descriptor_baseline is not None:
                c.eq(len(os.listdir("/dev/fd")), descriptor_baseline,
                     "mixed-root rejection closes every held descriptor")

            c.true(validation_calls >= 1,
                   "the swap occurs only after the initial committed validation")
            if not rejected:
                c.true(authorized is not None)
                c.eq(authorized.commitment, expected_commitment,
                     "returned commitment remains the held generation A")
                c.eq(authorized.recordings[0].cache_sha256, expected_cache_sha256,
                     "returned cache bytes remain the held generation A")
                c.eq(float(authorized.recordings[0].features_30hz[0, 0]),
                     expected_feature,
                     "returned features never come from swapped generation B")


def test_committed_mayo_authorizer_gates_all_live_manifest_sizes_before_read(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            targets = (
                fixture.output / "collection_manifest.json",
                fixture.output / "mayo_exposure_manifest.json",
                fixture.exposure,
            )
            original_read = builder.os.read
            for target in targets:
                original_payload = target.read_bytes()
                with target.open("wb") as handle:
                    handle.truncate(4 * 1024 * 1024 + 1)
                target.chmod(0o600)
                target_identity = (
                    int(target.stat().st_dev), int(target.stat().st_ino)
                )
                guarded_reads: list[int] = []

                def guarded_read(descriptor, count):
                    info = os.fstat(descriptor)
                    if (int(info.st_dev), int(info.st_ino)) == target_identity:
                        guarded_reads.append(descriptor)
                        raise AssertionError("oversized live manifest reached os.read")
                    return original_read(descriptor, count)

                builder.os.read = guarded_read
                try:
                    c.raises(
                        fixture.authorize,
                        ValueError,
                        f"{target.name} size is gated from fstat before reading",
                    )
                finally:
                    builder.os.read = original_read
                    target.write_bytes(original_payload)
                    target.chmod(0o600)
                c.eq(
                    guarded_reads,
                    [],
                    f"{target.name} oversized bytes are never read",
                )


def test_committed_mayo_generation_aggregate_budget_precedes_cache_read(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            cache = next((fixture.output / "mediapipe").glob("*.npz"))
            with cache.open("r+b") as handle:
                handle.truncate(128 * 1024 * 1024 + 1)
            cache.chmod(0o600)
            original_read = builder._read_regular_descriptor
            cache_reads: list[str] = []

            def reject_cache_read(*args, **kwargs):
                field = str(kwargs.get("field", ""))
                if "cache" in field.lower():
                    cache_reads.append(field)
                    raise AssertionError("aggregate-overflow cache bytes were read")
                return original_read(*args, **kwargs)

            builder._read_regular_descriptor = reject_cache_read
            try:
                c.raises(
                    fixture.authorize,
                    ValueError,
                    "shared 128 MiB generation budget fails before cache reads",
                )
            finally:
                builder._read_regular_descriptor = original_read
            c.eq(cache_reads, [], "aggregate overflow reaches no cache read")


def test_staged_mayo_generation_aggregate_budget_precedes_cache_validation(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging = _semantic_staging(
            root,
            ".mayo_ssl_cache.staging-aggregate-limit",
            b"s" * 32,
            include_arkit=True,
            include_exclusions=True,
        )
        cache = next((staging / "mediapipe").glob("*.npz"))
        with cache.open("r+b") as handle:
            handle.truncate(128 * 1024 * 1024 + 1)
        cache.chmod(0o600)
        original_validate = builder._validate_compact_cache
        validation_calls = 0

        def unexpected_validation(*_args, **_kwargs):
            nonlocal validation_calls
            validation_calls += 1
            raise AssertionError("aggregate-overflow staging reached cache validation")

        builder._validate_compact_cache = unexpected_validation
        try:
            c.raises(
                lambda: builder._validate_staging(staging, salt=b"s" * 32),
                ValueError,
                "shared staging budget fails before cache validation",
            )
        finally:
            builder._validate_compact_cache = original_validate
        c.eq(validation_calls, 0, "staging overflow reaches no cache parser")


def test_nonheld_mayo_manifests_are_size_gated_before_read(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        targets = (
            root / "staging" / "collection_manifest.json",
            root / "mayo_exposure_manifest.json",
        )
        original_read = builder.os.read
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.parent.chmod(0o700)
            with target.open("wb") as handle:
                handle.truncate(4 * 1024 * 1024 + 1)
            target.chmod(0o600)
            guarded_reads: list[int] = []

            def guarded_read(descriptor, count):
                guarded_reads.append(descriptor)
                raise AssertionError("oversized non-held manifest reached os.read")

            builder.os.read = guarded_read
            try:
                c.raises(
                    lambda value=target: builder._load_public_json(
                        value, "non-held Mayo manifest"
                    ),
                    ValueError,
                    f"{target.name} size is gated from fstat before reading",
                )
            finally:
                builder.os.read = original_read
            c.eq(
                guarded_reads,
                [],
                f"{target.name} non-held oversized bytes are never read",
            )


def test_output_parent_lock_cleanup_is_exception_safe(c: Check):
    for failure_point in ("unlock", "close"):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "mayo_ssl_cache"
            original_flock = builder.fcntl.flock
            original_close = builder.os.close
            lock_descriptors: list[int] = []
            close_attempts: list[int] = []
            failed = False

            def tracked_flock(descriptor, operation):
                nonlocal failed
                if operation == builder.fcntl.LOCK_EX | builder.fcntl.LOCK_NB:
                    lock_descriptors.append(descriptor)
                if failure_point == "unlock" and operation == builder.fcntl.LOCK_UN:
                    original_flock(descriptor, operation)
                    failed = True
                    raise OSError("synthetic unlock failure")
                return original_flock(descriptor, operation)

            def tracked_close(descriptor):
                nonlocal failed
                close_attempts.append(descriptor)
                original_close(descriptor)
                if failure_point == "close" and descriptor in lock_descriptors:
                    failed = True
                    raise OSError("synthetic close failure")

            builder.fcntl.flock = tracked_flock
            builder.os.close = tracked_close
            caught: BaseException | None = None
            try:
                try:
                    with builder.output_parent_lock(output):
                        raise RuntimeError("primary lock body failure")
                except BaseException as exc:  # inspect cleanup chaining
                    caught = exc
            finally:
                builder.os.close = original_close
                builder.fcntl.flock = original_flock
                for descriptor in lock_descriptors:
                    if descriptor not in close_attempts:
                        try:
                            original_close(descriptor)
                        except OSError:
                            pass
            c.true(failed, f"{failure_point} cleanup failure was injected")
            c.true(caught is not None, f"{failure_point} cleanup failure is surfaced")
            c.true(
                bool(lock_descriptors)
                and all(descriptor in close_attempts for descriptor in lock_descriptors),
                f"lock FD is closed even when {failure_point} fails",
            )
            c.true(
                caught is not None and _exception_chain_contains(
                    caught, RuntimeError, "primary lock body failure"
                ),
                f"{failure_point} cleanup preserves the primary exception chain",
            )


def test_output_parent_lock_rebinds_live_name_after_flock_before_yield(c: Check):
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        parent.chmod(0o700)
        output = parent / "mayo_ssl_cache"
        lock = parent / ".mayo_ssl_cache.lock"
        parked = parent / ".mayo_ssl_cache.lock.parked"
        lock.write_bytes(b"")
        lock.chmod(0o600)
        original_flock = builder.fcntl.flock
        replacement_descriptor: int | None = None
        swapped = False
        entered = False

        def split_brain_flock(descriptor, operation):
            nonlocal replacement_descriptor, swapped
            if operation == builder.fcntl.LOCK_EX | builder.fcntl.LOCK_NB and not swapped:
                lock.rename(parked)
                lock.write_bytes(b"")
                lock.chmod(0o600)
                replacement_descriptor = os.open(lock, os.O_RDWR)
                original_flock(
                    replacement_descriptor,
                    builder.fcntl.LOCK_EX | builder.fcntl.LOCK_NB,
                )
                swapped = True
            return original_flock(descriptor, operation)

        def enter_lock() -> None:
            nonlocal entered
            with builder.output_parent_lock(output):
                entered = True

        builder.fcntl.flock = split_brain_flock
        try:
            c.raises(
                enter_lock,
                ValueError,
                "post-flock live-name replacement fails before the lock body",
            )
        finally:
            builder.fcntl.flock = original_flock
            if replacement_descriptor is not None:
                try:
                    original_flock(replacement_descriptor, builder.fcntl.LOCK_UN)
                finally:
                    os.close(replacement_descriptor)
        c.true(swapped)
        c.true(not entered, "split-brain destination lock never enters the body")


def test_held_mayo_generation_attempts_every_fd_close_after_one_failure(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            original_close = builder.os.close
            expected_descriptors: list[int] = []
            close_attempts: list[int] = []
            fail_descriptor: int | None = None
            failed = False

            def tracked_close(descriptor):
                nonlocal failed
                close_attempts.append(descriptor)
                original_close(descriptor)
                if descriptor == fail_descriptor and not failed:
                    failed = True
                    raise OSError("synthetic held descriptor close failure")

            builder.os.close = tracked_close
            caught: BaseException | None = None
            try:
                try:
                    with builder._hold_committed_mayo_generation(
                        fixture.output, fixture.exposure
                    ) as held:
                        expected_descriptors = [
                            held.output_parent_descriptor,
                            held.output_descriptor,
                            held.collection_descriptor,
                            held.internal_exposure_descriptor,
                            held.media_descriptor,
                            held.arkit_descriptor,
                            held.external_parent_descriptor,
                            held.external_exposure_descriptor,
                            *(item.descriptor for item in held.media_files),
                            *(item.descriptor for item in held.arkit_files),
                        ]
                        fail_descriptor = held.arkit_descriptor
                        raise RuntimeError("primary held generation failure")
                except BaseException as exc:  # inspect cleanup chaining
                    caught = exc
            finally:
                builder.os.close = original_close
                for descriptor in expected_descriptors:
                    if descriptor not in close_attempts:
                        try:
                            original_close(descriptor)
                        except OSError:
                            pass
            c.true(failed, "one held descriptor close failure was injected")
            c.eq(
                set(close_attempts),
                set(expected_descriptors),
                "every held descriptor close is attempted after an intermediate failure",
            )
            c.true(
                caught is not None and _exception_chain_contains(
                    caught, RuntimeError, "primary held generation failure"
                ),
                "held descriptor cleanup preserves the primary exception chain",
            )


def test_committed_mayo_authorizer_live_recomputes_v3_and_rejects_transaction_state(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        salt_bytes = b"m" * 32
        staging = _semantic_staging(
            root, ".mayo_ssl_cache.staging-fixture", salt_bytes,
            include_arkit=True, include_exclusions=True,
        )
        output = root / "outputs" / "dynamic_landmark" / "pretraining" / "mayo_ssl_cache"
        exposure = root / "outputs" / "dynamic_landmark" / "mayo_exposure_manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        exposure.parent.mkdir(parents=True, exist_ok=True)
        exposure.parent.chmod(0o700)
        output.parent.chmod(0o700)
        os.replace(staging, output)
        exposure.write_bytes((output / "mayo_exposure_manifest.json").read_bytes())
        exposure.chmod(0o600)
        key = output.parent / ".mayo_ssl_hmac.key"
        key.write_bytes(salt_bytes)
        key.chmod(0o600)
        lock = output.parent / f".{output.name}.lock"
        lock.write_bytes(b"")
        lock.chmod(0o600)
        collection = json.loads((output / "collection_manifest.json").read_text())
        exposure_value = json.loads(exposure.read_text())
        counts = dict(collection["counts"])
        data = root / "data" / "livelinkface_data"
        exports = root / "data" / "mediapipe_out"
        data.mkdir(parents=True)
        exports.mkdir()
        inventory = builder.MayoInventory(
            data, exports, counts, (), (), (), (), (), (), (), (), (),
        )
        original_root = builder.PROJECT_ROOT
        original_frozen = builder.FROZEN_INVENTORY
        original_inventory = builder.inventory_mayo_sources
        original_manifest_builder = builder.build_public_manifests
        builder.PROJECT_ROOT = root
        builder.FROZEN_INVENTORY = counts
        builder.inventory_mayo_sources = lambda *_args, **_kwargs: inventory
        builder.build_public_manifests = lambda *_args, **_kwargs: (
            json.loads(json.dumps(collection)),
            json.loads(json.dumps(exposure_value)),
        )
        try:
            before = {
                path.relative_to(root): path.read_bytes()
                for path in (key, exposure, *(item for item in output.rglob("*") if item.is_file()))
            }
            authorized = builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            )
            c.eq(authorized.recording_count, 1)
            c.eq(authorized.arkit_count, 1)
            c.eq(len(authorized.recordings), 1,
                 "ARKit remains outside the main bridge recordings")
            c.eq(authorized.commitment["schema"],
                 "mayo_cache_generation_commitment_v3")
            c.eq({
                path.relative_to(root): path.read_bytes()
                for path in (key, exposure, *(item for item in output.rglob("*") if item.is_file()))
            }, before, "committed authorization does not recover or rewrite data")

            journal = output.parent / f".{output.name}.transaction.json"
            journal.write_text("unresolved", encoding="utf-8")
            journal.chmod(0o600)
            c.raises(lambda: builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            ), RuntimeError, "unresolved transaction state is rejected, never recovered")
            c.eq(journal.read_text(), "unresolved",
                 "read-only authorization leaves the unresolved journal untouched")
            journal.unlink()

            residue = output.parent / f".{output.name}.staging-interrupted"
            residue.mkdir()
            c.raises(lambda: builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            ), RuntimeError, "unresolved staging residue is rejected, never recovered")
            c.true(residue.is_dir(), "read-only authorization leaves residue untouched")
            residue.rmdir()

            cache = next((output / "mediapipe").glob("*.npz"))
            original_cache = cache.read_bytes()
            cache.write_bytes(original_cache[:-1] + bytes([original_cache[-1] ^ 1]))
            c.raises(lambda: builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            ), ValueError, "a changed main-cache byte invalidates live v3 authorization")
            cache.write_bytes(original_cache)
            key.chmod(0o640)
            c.raises(lambda: builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            ), ValueError, "the canonical Mayo key must remain exact mode 0600")
            key.chmod(0o600)

            collection_path = output / "collection_manifest.json"
            original_collection = collection_path.read_bytes()
            changed_collection = json.loads(original_collection)
            changed_collection["dataset"] = "forged-live-manifest"
            collection_path.write_text(json.dumps(changed_collection), encoding="utf-8")
            c.raises(lambda: builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            ), ValueError, "a changed committed manifest fails closed")
            collection_path.write_bytes(original_collection)
            collection_path.chmod(0o600)

            hardlink = output.parent / ".hardlinked-mayo-key"
            os.link(key, hardlink)
            c.raises(lambda: builder.authorize_committed_mayo_ssl_generation(
                data, exports, key, output, exposure,
            ), ValueError, "a multiply-linked canonical Mayo key is rejected")
            hardlink.unlink()

            lock.unlink()
            rejected = False
            try:
                builder.authorize_committed_mayo_ssl_generation(
                    data, exports, key, output, exposure,
                )
            except ValueError:
                rejected = True
            c.true(rejected, "read-only authorization requires the existing lock")
            c.true(not lock.exists(),
                   "read-only authorization never O_CREATs a missing lock")
        finally:
            builder.PROJECT_ROOT = original_root
            builder.FROZEN_INVENTORY = original_frozen
            builder.inventory_mayo_sources = original_inventory
            builder.build_public_manifests = original_manifest_builder


def test_committed_mayo_authorizer_missing_lock_is_read_only(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            fixture.lock.unlink()
            before = {
                path.relative_to(root): (
                    path.read_bytes(), path.stat().st_mode,
                    path.stat().st_mtime_ns,
                )
                for path in root.rglob("*") if path.is_file()
            }
            c.raises(
                fixture.authorize,
                ValueError,
                "read-only Mayo authorization requires the existing producer lock",
            )
            c.true(not fixture.lock.exists(), "authorization never O_CREATs a lock")
            c.eq({
                path.relative_to(root): (
                    path.read_bytes(), path.stat().st_mode,
                    path.stat().st_mtime_ns,
                )
                for path in root.rglob("*") if path.is_file()
            }, before, "a rejected authorization leaves every file unchanged")


def test_committed_mayo_authorizer_rechecks_live_root_classification(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            drifted = builder.MayoInventory(
                fixture.data, fixture.exports, fixture.counts,
                (), (), (), (), (), (), (), (), (Path("opaque-drift"),),
            )
            inventory_calls = 0

            def changing_inventory(*_args, **_kwargs):
                nonlocal inventory_calls
                inventory_calls += 1
                return fixture.inventory if inventory_calls == 1 else drifted

            def live_manifests(inventory, *_args, **_kwargs):
                collection = json.loads(json.dumps(fixture.collection))
                exposure = json.loads(json.dumps(fixture.exposure_value))
                if inventory is drifted:
                    collection["classification_integrity_id"] = "agg_" + "1" * 64
                    exposure["classification_integrity_id"] = "agg_" + "2" * 64
                return collection, exposure

            builder.inventory_mayo_sources = changing_inventory
            builder.build_public_manifests = live_manifests
            c.raises(
                fixture.authorize,
                ValueError,
                "a live-root classification change during authorization fails closed",
            )
            c.eq(inventory_calls, 2, "the live roots are audited again before return")


def test_committed_mayo_authorizer_holds_every_cache_mode_through_return(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _CommittedMayoAuthorizerFixture(root) as fixture:
            cache = next((fixture.output / "mediapipe").glob("*.npz"))
            inventory_calls = 0

            def mutate_after_final_cache_validation(*_args, **_kwargs):
                nonlocal inventory_calls
                inventory_calls += 1
                if inventory_calls == 2:
                    cache.chmod(0o666)
                return fixture.inventory

            builder.inventory_mayo_sources = mutate_after_final_cache_validation
            try:
                c.raises(
                    fixture.authorize,
                    ValueError,
                    "cache permission drift after validation fails before return",
                )
            finally:
                cache.chmod(0o600)
            c.eq(inventory_calls, 2)


if __name__ == "__main__":
    run_all("test_build_mayo_ssl_cache", dict(globals()))
