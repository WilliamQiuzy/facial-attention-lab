"""Fail-closed tests for the deidentified PalsyNet clinical23_v2 builder."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import sys
import tempfile
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
    load_dynamic_landmark_recording,
)

SCRIPT = ROOT / "scripts" / "build_palsynet_v2_windows.py"
SPEC = importlib.util.spec_from_file_location("build_palsynet_v2_windows", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load builder module")
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(index: int, label: str, source_sha256: str) -> dict[str, object]:
    return {
        "claim_unit": "video_held_out",
        "group_id": f"grp_{index:064x}",
        "identity_status": "unreviewed",
        "label": label,
        "recording_id": f"rec_{index:064x}",
        "source_sha256": source_sha256,
    }


def _manifest(records: list[dict[str, object]]) -> dict[str, object]:
    source_fingerprint = hashlib.sha256()
    for row in sorted(records, key=lambda item: (item["label"], item["source_sha256"])):
        source_fingerprint.update(
            f"{row['label']}:{row['source_sha256']}\n".encode("ascii")
        )
    pairs = [
        {
            "cosine": 0.5,
            "rank": rank,
            "recording_id_a": left["recording_id"],
            "recording_id_b": right["recording_id"],
        }
        for rank, (left, right) in enumerate(itertools.combinations(records, 2), start=1)
    ]
    return {
        "claim_unit": "video_held_out",
        "contact_sheet_sampling": {
            "raw_filename_text_burned_in": False,
            "representative_frame_offset": 16,
            "window_size_frames": 32,
            "windows_per_video": 4,
        },
        "contact_sheets": {
            "filenames": "opaque_ids_or_ranks_only",
            "ranked_pairs": 25,
            "recordings": 49,
            "storage": "local_ignored_output",
        },
        "counts": {"affected": 27, "ranked_pairs": 1176, "total": 49,
                   "unaffected": 22},
        "dataset": "PalsyNet",
        "fingerprints": {
            "bundle_provenance_sha256": "a" * 64,
            "embedding_collection_sha256": "b" * 64,
            "source_collection_sha256": source_fingerprint.hexdigest(),
        },
        "identity_review": {
            "group_override_applied": True,
            "manual_review_required": True,
            "reviewer_evidence_sha256": None,
            "status": "unreviewed",
        },
        "ranked_pairs": pairs,
        "recordings": records,
        "schema_version": "palsynet_identity_audit_v1",
    }


def _full_records(hashes: list[str] | None = None) -> list[dict[str, object]]:
    hashes = hashes or [_sha(f"video-{index}".encode()) for index in range(49)]
    rows = [
        _record(index + 1, "affected" if index < 27 else "unaffected", digest)
        for index, digest in enumerate(hashes)
    ]
    # The confirmed duplicate videos share one identity group.
    rows[1]["group_id"] = rows[0]["group_id"]
    return rows


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(_manifest(records), sort_keys=True))


def test_identity_manifest_is_exact_and_preserves_conservative_claim(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "identity_manifest.json"
        records = _full_records()
        _write_manifest(path, records)
        loaded = builder.load_identity_manifest(path)
        c.eq(len(loaded.by_source_sha256), 49, "all 49 hashes are join keys")
        c.eq(len({row.group_id for row in loaded.by_source_sha256.values()}), 48,
             "confirmed duplicate remains in one shared group")
        c.eq(loaded.claim_unit, "video_held_out",
             "unreviewed audit cannot silently become person held out")
        c.eq(loaded.identity_status, "unreviewed", "review status is preserved")

        bad_cases: list[list[dict[str, object]]] = []
        bad_cases.append(records[:-1])
        bad_cases.append(records + [_record(50, "unaffected", _sha(b"extra"))])
        swapped = [dict(row) for row in records]
        swapped[0]["label"] = "unaffected"
        bad_cases.append(swapped)
        duplicate_hash = [dict(row) for row in records]
        duplicate_hash[1]["source_sha256"] = duplicate_hash[0]["source_sha256"]
        bad_cases.append(duplicate_hash)
        duplicate_rec = [dict(row) for row in records]
        duplicate_rec[1]["recording_id"] = duplicate_rec[0]["recording_id"]
        bad_cases.append(duplicate_rec)
        cross_label = [dict(row) for row in records]
        cross_label[-1]["group_id"] = cross_label[0]["group_id"]
        bad_cases.append(cross_label)
        for index, rows in enumerate(bad_cases):
            candidate = Path(td) / f"bad_{index}.json"
            _write_manifest(candidate, rows)
            c.raises(lambda candidate=candidate: builder.load_identity_manifest(candidate),
                     ValueError, f"bad manifest case {index} fails closed")


def test_identity_manifest_rejects_fingerprint_and_ranked_pair_corruption(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        records = _full_records()
        cases = []
        wrong_fingerprint = _manifest(records)
        wrong_fingerprint["fingerprints"]["source_collection_sha256"] = "f" * 64
        cases.append(wrong_fingerprint)
        self_pair = _manifest(records)
        self_pair["ranked_pairs"][0]["recording_id_b"] = self_pair["ranked_pairs"][0]["recording_id_a"]
        cases.append(self_pair)
        duplicate_pair = _manifest(records)
        duplicate_pair["ranked_pairs"][1]["recording_id_a"] = duplicate_pair["ranked_pairs"][0]["recording_id_a"]
        duplicate_pair["ranked_pairs"][1]["recording_id_b"] = duplicate_pair["ranked_pairs"][0]["recording_id_b"]
        cases.append(duplicate_pair)
        bad_cosine = _manifest(records)
        bad_cosine["ranked_pairs"][0]["cosine"] = 1.01
        cases.append(bad_cosine)
        for index, payload in enumerate(cases):
            path = root / f"corrupt_{index}.json"
            path.write_text(json.dumps(payload))
            c.raises(lambda path=path: builder.load_identity_manifest(path), ValueError,
                     f"identity corruption case {index} fails closed")


def test_source_join_uses_content_hash_not_filename(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "data"
        (root / "affected").mkdir(parents=True)
        (root / "unaffected").mkdir()
        payloads = [f"video-{index}".encode() for index in range(49)]
        paths = []
        for index, payload in enumerate(payloads):
            label = "affected" if index < 27 else "unaffected"
            # Names deliberately carry the opposite-looking class token.
            name = f"unaffected-looking-{48-index}.mp4" if label == "affected" else f"affected-looking-{48-index}.mp4"
            path = root / label / name
            path.write_bytes(payload)
            paths.append(path)
        manifest_path = Path(td) / "identity_manifest.json"
        _write_manifest(manifest_path, _full_records([_sha(payload) for payload in payloads]))
        identity = builder.load_identity_manifest(manifest_path)
        first = builder.enumerate_source_videos(root, identity)
        c.eq(len(first), 49, "all source hashes join exactly once")
        by_hash = {item.source_sha256: item.binding.recording_id for item in first}

        for index, path in enumerate(paths):
            renamed = path.with_name(f"renamed completely {index}.mp4")
            path.rename(renamed)
        second = builder.enumerate_source_videos(root, identity)
        c.eq({item.source_sha256: item.binding.recording_id for item in second}, by_hash,
             "renaming files cannot change rec/group identity")


class FakeCapture:
    def __init__(self, frame_count: float = 160.0, fps: float = 25.0,
                 fail_seek: bool = False, fail_decode_at: int | None = None,
                 position_offset: float = 0.0):
        self.frame_count = frame_count
        self.fps = fps
        self.fail_seek = fail_seek
        self.fail_decode_at = fail_decode_at
        self.position_offset = position_offset
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 8.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 6.0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position) + self.position_offset
        raise AssertionError(prop)

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(prop)
        if self.fail_seek:
            return False
        self.position = int(value)
        return True

    def read(self):
        if self.fail_decode_at == self.position:
            return False, None
        frame = np.full((6, 8, 3), self.position % 255, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class FakeExtractor:
    feature_schema = DYNAMIC_FEATURE_SCHEMA
    feature_names = list(DYNAMIC_FEATURE_NAMES)
    capture_mirrored = None

    def __init__(self, miss_indices: set[int] | None = None):
        self.miss_indices = miss_indices or set()
        self.seen: list[int] = []

    def extract_frame_with_nuisance(self, frame):
        index = int(frame[0, 0, 0])
        self.seen.append(index)
        if index in self.miss_indices:
            return None, None
        features = np.zeros(95, dtype=np.float32)
        features[0] = index
        features[-23:] = np.arange(23, dtype=np.float32) + index / 1000.0
        return features, {"face_scale": 0.2 + index / 10000.0,
                          "eye_line_roll_degrees": index / 100.0}

    def _frame_features_with_nuisance(self, _frame):
        raise AssertionError("builder must use the public same-detection API")


def _source(tmp: Path, digest: str | None = None):
    path = tmp / "private-human-name.mp4"
    path.write_bytes(b"synthetic video")
    digest = digest or _sha(path.read_bytes())
    binding = builder.IdentityBinding(
        source_sha256=digest,
        recording_id="rec_" + "1" * 64,
        group_id="grp_" + "2" * 64,
        label="affected",
        identity_status="unreviewed",
        claim_unit="video_held_out",
    )
    return builder.SourceVideo(path=path, source_sha256=digest, binding=binding)


def test_locked_decode_uses_exact_indices_timestamps_and_schema(c: Check):
    with tempfile.TemporaryDirectory() as td:
        capture = FakeCapture(frame_count=160.0, fps=25.0)
        extractor = FakeExtractor()
        result = builder.extract_source_video(
            _source(Path(td)), extractor, capture_factory=lambda _path: capture)
        expected_indices = np.stack([
            np.arange(start, start + 32, dtype=np.int64)
            for start in (0, 42, 85, 128)
        ])
        c.true(bool(np.array_equal(result.source_frame_indices, expected_indices)),
               "exact frozen source indices are decoded")
        c.true(bool(np.allclose(result.timestamps, expected_indices / 25.0)),
               "timestamps are source index divided by finite FPS")
        c.eq(result.features.shape, (4, 32, 95), "exact fused cache shape")
        c.eq(tuple(extractor.feature_names), DYNAMIC_FEATURE_NAMES,
             "exact 95-column order is enforced")
        c.eq(result.coverage, 1.0, "all fake detections are retained")
        c.true(capture.released, "capture is released")


def test_decode_and_metadata_failures_are_recording_fatal(c: Check):
    with tempfile.TemporaryDirectory() as td:
        source = _source(Path(td))
        failures = (
            FakeCapture(fps=0.0),
            FakeCapture(fps=float("nan")),
            FakeCapture(frame_count=127.0),
            FakeCapture(frame_count=160.5),
            FakeCapture(fail_seek=True),
            FakeCapture(fail_decode_at=43),
            FakeCapture(position_offset=0.5),
        )
        for index, capture in enumerate(failures):
            c.raises(
                lambda capture=capture: builder.extract_source_video(
                    source, FakeExtractor(), capture_factory=lambda _path: capture
                ),
                builder.RecordingExtractionError,
                f"decode failure case {index} fails the whole recording",
            )


def test_unopened_capture_is_still_released(c: Check):
    with tempfile.TemporaryDirectory() as td:
        capture = FakeCapture()
        capture.isOpened = lambda: False
        c.raises(
            lambda: builder.extract_source_video(
                _source(Path(td)), FakeExtractor(),
                capture_factory=lambda _path: capture,
            ),
            builder.RecordingExtractionError,
            "unopened source is a recording-level failure",
        )
        c.true(capture.released, "unopened OpenCV capture is still released")


def test_misses_are_zero_masked_and_coverage_gate_is_exact(c: Check):
    with tempfile.TemporaryDirectory() as td:
        source = _source(Path(td))
        selected = list(range(128))
        # For the 128-frame minimum, all source frames are selected exactly once.
        for misses, passes in ((set(selected[:13]), False), (set(selected[:12]), True)):
            result = builder.extract_source_video(
                source,
                FakeExtractor(misses),
                capture_factory=lambda _path: FakeCapture(frame_count=128.0, fps=30.0),
            )
            if passes:
                builder.validate_retained_recording(result)
                c.eq(int(result.valid_mask.sum()), 116, "116 detections pass")
                c.true(bool((result.features[~result.valid_mask] == 0).all()),
                       "misses are canonical zero rows")
            else:
                c.raises(lambda result=result: builder.validate_retained_recording(result),
                         ValueError, "115 detections fail the 90 percent gate")


def test_payload_round_trip_uses_only_exact_dynamic_fields(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        result = builder.extract_source_video(
            _source(root), FakeExtractor(),
            capture_factory=lambda _path: FakeCapture(frame_count=128.0, fps=30.0),
        )
        path = root / f"{result.binding.recording_id}.npz"
        builder.write_validated_recording_cache(path, result)
        loaded = load_dynamic_landmark_recording(path)
        with np.load(path, allow_pickle=False) as saved:
            c.eq(set(saved.files), builder.DYNAMIC_CACHE_FIELDS,
                 "cache contains the exact loader field set")
        c.eq(loaded.recording_id, result.binding.recording_id,
             "temp write is reread through the production loader")
        c.eq(path.name, result.binding.recording_id + ".npz",
             "cache filename exposes only opaque recording id")


def _outcome(index: int, label: str, group: int, *, coverage: float = 1.0,
             varied: bool = True):
    return SimpleNamespace(
        binding=SimpleNamespace(
            recording_id=f"rec_{index:064x}",
            group_id=f"grp_{group:064x}",
            label=label,
        ),
        coverage=coverage,
        landmark_varied=varied,
    )


def test_collection_gate_checks_retention_groups_coverage_and_variation(c: Check):
    records = [
        _outcome(index + 1, "affected" if index < 27 else "unaffected", index + 1)
        for index in range(47)
    ]
    # Two same-label recordings represent the confirmed shared person.
    records[1].binding.group_id = records[0].binding.group_id
    builder.validate_collection_gate(records, expected_total=49)
    c.raises(lambda: builder.validate_collection_gate(records[:-1], expected_total=49),
             ValueError, "fewer than 47 retained recordings fails")
    low_coverage = list(records)
    low_coverage[0] = _outcome(1, "affected", 1, coverage=0.89)
    c.raises(lambda: builder.validate_collection_gate(low_coverage, expected_total=49),
             ValueError, "every retained recording must pass coverage")
    low_variation = list(records)
    for index in range(3):
        low_variation[index] = _outcome(index + 1, "affected", index + 1, varied=False)
    c.raises(lambda: builder.validate_collection_gate(low_variation, expected_total=49),
             ValueError, "less than 95 percent landmark variation fails")
    too_few_control_groups = [
        _outcome(index + 1, "affected" if index < 42 else "unaffected",
                 index + 1 if index < 42 else 100 + (index % 4))
        for index in range(47)
    ]
    c.raises(lambda: builder.validate_collection_gate(
        too_few_control_groups, expected_total=49), ValueError,
        "each label needs five groups for five-fold group CV")


def test_variation_uses_only_valid_last_23_columns(c: Check):
    features = np.zeros((4, 32, 95), dtype=np.float32)
    mask = np.ones((4, 32), dtype=bool)
    features[..., 0] = np.arange(128, dtype=np.float32).reshape(4, 32)
    varied, statistic = builder.landmark_variation(features, mask)
    c.true(not varied and statistic == 0.0,
           "blendshape changes cannot satisfy landmark variation")
    features[0, 1, -1] = 0.01
    varied, statistic = builder.landmark_variation(features, mask)
    c.true(varied and statistic > 0.0, "valid landmark change is detected")
    mask[0, 1] = False
    varied, statistic = builder.landmark_variation(features, mask)
    c.true(not varied and statistic == 0.0, "masked landmark change is ignored")


def test_variation_never_bridges_noncontiguous_window_offsets(c: Check):
    features = np.zeros((4, 32, 95), dtype=np.float32)
    mask = np.ones((4, 32), dtype=bool)
    for window_index in range(4):
        features[window_index, :, -23:] = float(window_index)
    varied, statistic = builder.landmark_variation(features, mask)
    c.true(not varied and statistic == 0.0,
           "constant windows cannot gain variation from between-window offsets")
    features[2, 10, -1] += 0.01
    varied, statistic = builder.landmark_variation(features, mask)
    c.true(varied and statistic > 0.0,
           "one actual within-window landmark change is sufficient")


def _frozen_frame_counts() -> list[int]:
    # 49 recordings, sum 177,511, minimum 172.
    return [172] + [3695] * 27 + [3694] * 21


def test_frozen_corpus_metadata_preflight_matches_audited_inventory(c: Check):
    counts = _frozen_frame_counts()
    sources = [SimpleNamespace(path=Path(f"opaque-{index}.mp4"))
               for index in range(49)]
    captures: list[FakeCapture] = []

    def factory(path):
        index = int(Path(path).stem.split("-")[-1])
        capture = FakeCapture(frame_count=float(counts[index]), fps=30.0)
        captures.append(capture)
        return capture

    summary = builder.preflight_corpus_metadata(sources, capture_factory=factory)
    c.eq(summary["recordings"], 49, "exact audited source count")
    c.eq(summary["total_frames"], 177_511, "exact audited frame total")
    c.eq(summary["minimum_frames"], 172, "exact audited minimum")
    c.true(abs(summary["duration_minutes"] - 98.61722222222221) < 1e-12,
           "duration is derived from exact frame counts and FPS")
    c.true(all(capture.released for capture in captures),
           "metadata captures are always released")


def test_frozen_corpus_metadata_preflight_rejects_fps_count_minimum_and_aggregate(c: Check):
    baseline = _frozen_frame_counts()
    sources = [SimpleNamespace(path=Path(f"opaque-{index}.mp4"))
               for index in range(49)]

    def run(counts, fps_values):
        def factory(path):
            index = int(Path(path).stem.split("-")[-1])
            return FakeCapture(frame_count=float(counts[index]), fps=float(fps_values[index]))
        return builder.preflight_corpus_metadata(sources, capture_factory=factory)

    wrong_fps = [30.0] * 49
    wrong_fps[0] = 29.999
    c.raises(lambda: run(baseline, wrong_fps), ValueError,
             "non-30-Hz recording fails strict corpus preflight")

    nonintegral = list(baseline)
    nonintegral[1] = 3694.5
    c.raises(lambda: run(nonintegral, [30.0] * 49), ValueError,
             "nonintegral recording frame count fails")

    wrong_minimum = list(baseline)
    wrong_minimum[0] = 173
    wrong_minimum[1] -= 1
    c.raises(lambda: run(wrong_minimum, [30.0] * 49), ValueError,
             "wrong audited minimum fails even when aggregate is unchanged")

    wrong_aggregate = list(baseline)
    wrong_aggregate[-1] += 1
    c.raises(lambda: run(wrong_aggregate, [30.0] * 49), ValueError,
             "wrong frame total and derived duration fail")


def test_all_detector_misses_are_a_recording_exclusion_not_layout_typeerror(c: Check):
    with tempfile.TemporaryDirectory() as td:
        extractor = object.__new__(builder.MediaPipeFeatureExtractor)
        extractor.landmark_features = "clinical23"
        extractor.capture_mirrored = None
        extractor._bs_names = None
        extractor._pairs = None
        extractor.extract_frame_with_nuisance = lambda _frame: (None, None)
        try:
            builder.extract_source_video(
                _source(Path(td)),
                extractor,
                capture_factory=lambda _path: FakeCapture(
                    frame_count=128.0, fps=30.0
                ),
            )
        except builder.RecordingExtractionError as exc:
            c.eq(str(exc), "no_valid_detections",
                 "all misses have a stable deidentified exclusion reason")
        except BaseException as exc:  # noqa: BLE001
            raise AssertionError(
                f"all misses leaked {type(exc).__name__} instead of recording error"
            ) from exc
        else:
            raise AssertionError("all detector misses must exclude the recording")


def test_output_path_containment_symlinks_and_cli_contract(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "palsynet" / "data"
        data_root.mkdir(parents=True)
        model = root / "face_landmarker.task"
        model.write_bytes(b"model")
        manifest = root / "identity_manifest.json"
        manifest.write_text("{}")
        canonical = data_root.parent / "derived" / "clinical23_v2_windows"
        builder.validate_cli_paths(data_root, model, manifest, canonical)
        c.raises(lambda: builder.validate_cli_paths(
            data_root, model, manifest, root / "wrong-output"), ValueError,
            "output root is exact and non-configurable")
        outside = root / "outside"
        outside.mkdir()
        linked = root / "linked-model.task"
        linked.symlink_to(model)
        c.raises(lambda: builder.validate_cli_paths(
            data_root, linked, manifest, canonical), ValueError,
            "symlink inputs are rejected")
        real_component = root / "real-component"
        real_component.mkdir()
        nested_model = real_component / "nested.task"
        nested_model.write_bytes(b"nested model")
        linked_component = root / "linked-component"
        linked_component.symlink_to(real_component, target_is_directory=True)
        c.raises(lambda: builder.validate_cli_paths(
            data_root, linked_component / "nested.task", manifest, canonical),
            ValueError, "intermediate symlink traversal is rejected")
        linked_output = data_root.parent / "derived"
        linked_output.symlink_to(outside, target_is_directory=True)
        c.raises(lambda: builder.validate_cli_paths(
            data_root, model, manifest, canonical), ValueError,
            "output parent symlink traversal is rejected")

    parser = builder._parser()
    options = {action.dest for action in parser._actions}
    c.true({"data_root", "model_path", "identity_manifest", "output_root"} <= options,
           "four required CLI paths are explicit")
    c.true(not ({"threshold", "window_len", "n_windows", "coverage"} & options),
           "frozen thresholds and windows are not CLI tunables")


def test_staged_validation_is_exact_deidentified_and_temporally_bound_to_npz(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging = root / ".clinical23_v2_windows.staging-test"
        staging.mkdir()
        result = builder.extract_source_video(
            _source(root), FakeExtractor(),
            capture_factory=lambda _path: FakeCapture(frame_count=160.0, fps=25.0),
        )
        builder.write_validated_recording_cache(
            staging / f"{result.binding.recording_id}.npz", result
        )
        manifest = {
            "schema_version": "palsynet_clinical23_v2_windows_v1",
            "dataset": "PalsyNet",
            "claim_unit": "video_held_out",
            "identity_status": "unreviewed",
            "records": [builder._record_manifest(result)],
            "excluded": [],
        }
        manifest_path = staging / "collection_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        builder.validate_staged_file_set(staging, manifest)

        temporal_tampers = {
            "window_starts": [0, 41, 85, 128],
            "frames_per_window": 31,
            "timestamp_unit": "milliseconds",
            "fps": 24.0,
            "source_frame_count": 159,
        }
        for field, value in temporal_tampers.items():
            tampered = json.loads(json.dumps(manifest))
            tampered["records"][0][field] = value
            manifest_path.write_text(json.dumps(tampered))
            c.raises(
                lambda tampered=tampered: builder.validate_staged_file_set(
                    staging, tampered
                ),
                ValueError,
                f"manifest {field} tamper cannot disagree with staged NPZ",
            )
        manifest_path.write_text(json.dumps(manifest))
        (staging / "stale.npz").write_bytes(b"stale")
        c.raises(lambda: builder.validate_staged_file_set(staging, manifest),
                 ValueError, "stale mixed-generation cache is rejected")
        (staging / "stale.npz").unlink()
        private_manifest = dict(manifest)
        private_manifest["private_path"] = "/private/human-name.mp4"
        manifest_path.write_text(json.dumps(private_manifest))
        c.raises(lambda: builder.validate_staged_file_set(staging, private_manifest),
                 ValueError, "manifest paths and names are rejected")


def test_snapshot_rejects_enumeration_toctou_and_detects_producer_drift(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = _source(root)
        model = root / "model.task"
        identity = root / "identity.json"
        producer = root / "producer.py"
        model.write_bytes(b"model")
        identity.write_bytes(b"identity")
        producer.write_bytes(b"producer-v1")

        source.path.write_bytes(b"mutated after enumeration")
        c.raises(lambda: builder.snapshot_provenance(
            [source], model, identity, producer_paths={"builder": producer}),
            ValueError, "snapshot cannot adopt a post-enumeration source digest")

        source = _source(root)
        snapshot = builder.snapshot_provenance(
            [source], model, identity, producer_paths={"builder": producer}
        )
        builder.assert_provenance_unchanged(snapshot)
        for path in (source.path, model, identity, producer):
            original = path.read_bytes()
            path.write_bytes(original + b"drift")
            c.raises(lambda snapshot=snapshot: builder.assert_provenance_unchanged(snapshot),
                     ValueError, "source/model/identity/producer drift blocks promotion")
            path.write_bytes(original)


def test_dependency_provenance_requires_exactly_one_opencv_distribution(c: Check):
    candidates = tuple(builder.OPENCV_DISTRIBUTIONS)

    def resolver_for(installed):
        versions = {
            "numpy": "1.26.4",
            "mediapipe": "0.10.35",
            "torch": "2.2.1",
            **installed,
        }
        def resolve(name):
            if name not in versions:
                raise builder.importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        return resolve

    c.raises(lambda: builder._dependency_versions(
        version_resolver=resolver_for({})), ValueError,
        "zero OpenCV wheels is ambiguous provenance")
    c.raises(lambda: builder._dependency_versions(version_resolver=resolver_for({
        candidates[0]: "4.8.1.78", candidates[1]: "4.8.1.78",
    })), ValueError, "multiple OpenCV wheels is ambiguous provenance")
    for candidate in candidates:
        versions = builder._dependency_versions(
            version_resolver=resolver_for({candidate: "4.8.1.78"})
        )
        c.eq(versions["opencv"], f"{candidate}==4.8.1.78",
             f"single installed wheel {candidate} is named exactly")
        c.true(all("unknown" not in value for value in versions.values()),
               "required dependency provenance never falls back to unknown")


def test_dependency_provenance_requires_exact_torch_distribution(c: Check):
    installed = {
        "numpy": "1.26.4",
        "mediapipe": "0.10.35",
        "opencv-python": "4.8.1.78",
    }

    def resolver_with(torch_value):
        versions = dict(installed)
        if torch_value is not None:
            versions["torch"] = torch_value

        def resolve(name):
            if name not in versions:
                raise builder.importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        return resolve

    c.raises(lambda: builder._dependency_versions(
        version_resolver=resolver_with(None)), ValueError,
        "missing torch distribution fails closed")
    c.raises(lambda: builder._dependency_versions(
        version_resolver=resolver_with("")), ValueError,
        "empty torch version fails closed")
    c.raises(lambda: builder._dependency_versions(
        version_resolver=resolver_with("unknown")), ValueError,
        "unknown torch version fails closed")
    versions = builder._dependency_versions(
        version_resolver=resolver_with("2.2.1")
    )
    c.eq(versions["torch"], "torch==2.2.1",
         "torch import closure is recorded as exact distribution provenance")


def test_run_builder_constructor_failure_and_error_close_leave_no_staging(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / "palsynet" / "data"
        data.mkdir(parents=True)
        model = root / "model.task"
        identity_path = root / "identity.json"
        model.write_bytes(b"model")
        identity_path.write_bytes(b"identity")
        output = data.parent / "derived" / "clinical23_v2_windows"
        empty_source_fingerprint = hashlib.sha256().hexdigest()
        identity = SimpleNamespace(
            manifest_sha256=builder.sha256_file(identity_path),
            fingerprints={"source_collection_sha256": empty_source_fingerprint},
            claim_unit="video_held_out",
            identity_status="unreviewed",
        )
        snapshot = SimpleNamespace(
            identity_manifest=(identity_path, identity.manifest_sha256),
        )
        originals = {
            name: getattr(builder, name)
            for name in (
                "load_identity_manifest", "enumerate_source_videos",
                "preflight_corpus_metadata", "snapshot_provenance",
            )
        }
        builder.load_identity_manifest = lambda _path: identity
        builder.enumerate_source_videos = lambda _data, _identity: ()
        builder.preflight_corpus_metadata = lambda _sources, **_kwargs: {}
        builder.snapshot_provenance = lambda *_args, **_kwargs: snapshot
        try:
            def constructor_failure(**_kwargs):
                raise RuntimeError("synthetic constructor failure")

            c.raises(lambda: builder.run_builder(
                data, model, identity_path, output,
                extractor_factory=constructor_failure,
            ), RuntimeError, "extractor constructor failure is surfaced")
            c.true(not any(output.parent.glob(f".{output.name}.staging-*")),
                   "constructor failure creates no staging generation")

            extractor = SimpleNamespace(closed=False)
            extractor.close = lambda: setattr(extractor, "closed", True)
            c.raises(lambda: builder.run_builder(
                data, model, identity_path, output,
                extractor_factory=lambda **_kwargs: extractor,
            ), ValueError, "empty synthetic collection fails after construction")
            c.true(extractor.closed, "extractor closes on builder error")
            c.true(not any(output.parent.glob(f".{output.name}.staging-*")),
                   "builder error removes its staging generation")
        finally:
            for name, value in originals.items():
                setattr(builder, name, value)


def test_managed_extractor_closes_on_success_and_exception(c: Check):
    for fail_inside in (False, True):
        extractor = SimpleNamespace(closed=False)
        extractor.close = lambda extractor=extractor: setattr(extractor, "closed", True)
        try:
            with builder.managed_extractor(
                lambda **_kwargs: extractor,
                model_path=Path("opaque-model.task"),
            ) as observed:
                c.true(observed is extractor, "managed extractor yields constructed instance")
                if fail_inside:
                    raise RuntimeError("synthetic extraction error")
        except RuntimeError:
            if not fail_inside:
                raise
        c.true(extractor.closed, "extractor closes on success and exception")


def test_full_synthetic_run_builder_promotes_49_caches_and_closes(c: Check):
    """Repository regression for the complete successful transactional lifecycle."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / "palsynet" / "data"
        (data / "affected").mkdir(parents=True)
        (data / "unaffected").mkdir()
        paths: list[Path] = []
        hashes: list[str] = []
        for index in range(49):
            label = "affected" if index < 27 else "unaffected"
            path = data / label / f"source_{index:02d}.mp4"
            payload = f"synthetic-video-{index}".encode("ascii")
            path.write_bytes(payload)
            paths.append(path)
            hashes.append(_sha(payload))

        identity_path = root / "identity_manifest.json"
        _write_manifest(identity_path, _full_records(hashes))
        model = root / "face_landmarker.task"
        model.write_bytes(b"synthetic model")
        output = data.parent / "derived" / "clinical23_v2_windows"
        counts = _frozen_frame_counts()
        path_to_index = {str(path): index for index, path in enumerate(paths)}
        captures: list[FakeCapture] = []

        def capture_factory(path):
            capture = FakeCapture(
                frame_count=float(counts[path_to_index[str(Path(path))]]),
                fps=30.0,
            )
            captures.append(capture)
            return capture

        extractors = []

        class LifecycleExtractor(FakeExtractor):
            def __init__(self, **_kwargs):
                super().__init__()
                self.closed = False
                extractors.append(self)

            def close(self):
                self.closed = True

        original_dependencies = builder._dependency_versions
        builder._dependency_versions = lambda: {
            "python": "python==3.10.0",
            "numpy": "numpy==1.26.4",
            "mediapipe": "mediapipe==0.10.35",
            "torch": "torch==2.2.1",
            "opencv": "opencv-python==4.8.1.78",
        }
        try:
            manifest = builder.run_builder(
                data,
                model,
                identity_path,
                output,
                extractor_factory=LifecycleExtractor,
                capture_factory=capture_factory,
            )
        finally:
            builder._dependency_versions = original_dependencies

        c.eq(manifest["counts"]["retained"], 49,
             "complete synthetic corpus retains all recordings")
        c.eq(manifest["counts"]["excluded"], 0,
             "complete synthetic corpus has no exclusions")
        c.eq(len(list(output.glob("rec_*.npz"))), 49,
             "atomic output contains exactly 49 opaque caches")
        c.true((output / "collection_manifest.json").is_file(),
               "complete manifest is promoted with the caches")
        c.true(len(extractors) == 1 and extractors[0].closed,
               "single reused extractor closes after successful promotion")
        c.true(all(capture.released for capture in captures),
               "preflight and extraction captures are all released")
        c.true(not any(output.parent.glob(f".{output.name}.staging-*")),
               "successful lifecycle leaves no staging directory")
        c.true(not any(output.parent.glob(f".{output.name}.backup-*")),
               "successful lifecycle leaves no backup directory")
        manifest_text = (output / "collection_manifest.json").read_text()
        c.true(all(path.name not in manifest_text for path in paths),
               "promoted manifest contains no raw source filename")


def test_output_lock_recovery_and_concurrent_holder_fail_closed(c: Check):
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        output = parent / "clinical23_v2_windows"

        def contend():
            with builder.output_parent_lock(output):
                pass

        with builder.output_parent_lock(output):
            c.raises(contend, RuntimeError, "concurrent builder cannot enter cleanup")

        backup = parent / ".clinical23_v2_windows.backup-only"
        backup.mkdir()
        (backup / "old.txt").write_text("old")
        with builder.output_parent_lock(output):
            builder.recover_interrupted_generations(output)
        c.eq((output / "old.txt").read_text(), "old",
             "single interrupted backup restores when output is absent")

        output.rename(parent / ".clinical23_v2_windows.backup-a")
        second = parent / ".clinical23_v2_windows.backup-b"
        second.mkdir()
        with builder.output_parent_lock(output):
            c.raises(lambda: builder.recover_interrupted_generations(output),
                     ValueError, "multiple backups without output are ambiguous")
        c.true((parent / ".clinical23_v2_windows.backup-a").exists() and second.exists(),
               "ambiguous backups are preserved for manual recovery")


def test_atomic_promotion_restores_old_generation_and_cleans_stale_staging(c: Check):
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        output = parent / "clinical23_v2_windows"
        output.mkdir()
        (output / "old.txt").write_text("old")
        stale = parent / ".clinical23_v2_windows.staging-stale"
        stale.mkdir()
        (stale / "stale.txt").write_text("stale")
        with builder.output_parent_lock(output):
            builder.recover_interrupted_generations(output)
        c.true(not stale.exists(), "stale real staging trees are removed")

        staging = parent / ".clinical23_v2_windows.staging-new"
        staging.mkdir()
        (staging / "new.txt").write_text("new")
        original_replace = builder.os.replace
        calls = {"count": 0}

        def fail_second_replace(source, destination):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("synthetic promotion failure")
            return original_replace(source, destination)

        builder.os.replace = fail_second_replace
        try:
            with builder.output_parent_lock(output):
                c.raises(lambda: builder.promote_generation(staging, output), OSError,
                         "promotion failure is surfaced")
        finally:
            builder.os.replace = original_replace
        c.eq((output / "old.txt").read_text(), "old",
             "prior complete generation is restored")
        c.true(not any(parent.glob(".clinical23_v2_windows.backup-*")),
               "rollback leaves no backup debris")


if __name__ == "__main__":
    run_all("test_build_palsynet_v2_windows", dict(globals()))
