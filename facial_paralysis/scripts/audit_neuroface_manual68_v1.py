#!/usr/bin/env python3
"""Audit MediaPipe semantic geometry against all NeuroFace manual 68-point frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_neuroface_external_v1 import _json  # noqa: E402
from scripts.run_neuroface_motion_pretrain_v1 import _write_no_overwrite_json  # noqa: E402
from src.datasets.neuroface_external_v1 import parse_landmark_text  # noqa: E402
from src.evaluation.neuroface_manual68_audit_v1 import (  # noqa: E402
    build_manual68_audit_report,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402
from src.preprocessing.openface68_semantic import openface68_to_semantic23  # noqa: E402
from src.preprocessing.semantic_landmarks import (  # noqa: E402
    clinical23_v2_to_semantic23,
)


ARCHIVES = {
    "als": (
        "als_videos", "archive/als/Videos.zip",
        "als_landmarks", "archive/als/Landmarks_gt.zip",
    ),
    "healthy_control": (
        "healthy_control_videos", "archive/healthy_controls/Videos.zip",
        "healthy_control_landmarks",
        "archive/healthy_controls/Landmarks_gt_and_VideoInfoFile_HC.zip",
    ),
    "post_stroke": (
        "post_stroke_videos", "archive/stroke/Videos_and_metadata.zip",
        "post_stroke_landmarks", "archive/stroke/Landmarks_gt.zip",
    ),
}
OUTPUT_RELATIVE = "outputs/neuroface_manual68_audit_v1/report.json"
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "evaluation" / "neuroface_manual68_audit_v1.py",
    PROJECT_ROOT / "src" / "preprocessing" / "openface68_semantic.py",
    PROJECT_ROOT / "src" / "preprocessing" / "semantic_landmarks.py",
    PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
    PROJECT_ROOT / "src" / "preprocessing" / "action_bundle.py",
    PROJECT_ROOT / "src" / "datasets" / "neuroface_external_v1.py",
)


def _sha256_file(path: Path) -> str:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("audit input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_sha256(values: dict[str, str]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256() -> str:
    return _collection_sha256({
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in _IMPLEMENTATION_FILES
    })


def _open_pinned_zip(path: Path, expected_sha256: str) -> zipfile.ZipFile:
    if _sha256_file(path) != expected_sha256:
        raise ValueError("NeuroFace archive differs from the private manifest")
    archive = zipfile.ZipFile(path)
    names: set[str] = set()
    try:
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if (
                not info.filename or pure.is_absolute() or "\\" in info.filename
                or "\x00" in info.filename
                or any(part in {"", ".", ".."} for part in pure.parts)
                or info.filename in names or info.flag_bits & 0x1
                or mode == stat.S_IFLNK
            ):
                raise ValueError("NeuroFace archive member contract is unsafe")
            names.add(info.filename)
    except BaseException:
        archive.close()
        raise
    return archive


def _members_by_stem(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(suffix):
            continue
        stem = PurePosixPath(info.filename).stem.lower()
        if stem in output:
            raise ValueError("archive contains duplicate relevant stems")
        output[stem] = info.filename
    return output


def _private_video_rows(private: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = private.get("records")
    if (
        private.get("schema_version") != "neuroface_external_private_manifest_v1"
        or not isinstance(rows, list) or len(rows) != 261
        or private.get("counts", {}).get("annotated_frames") != 3306
    ):
        raise ValueError("manual68 audit requires the frozen complete private manifest")
    output = {str(row["video_sha256"]): row for row in rows}
    if len(output) != 261:
        raise ValueError("private NeuroFace video hashes must be unique")
    return output


def _write_video(payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".manual68-video.", suffix=".avi")
    path = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--mediapipe-model", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    output = PROJECT_ROOT / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FileExistsError("manual68 audit report already exists")
    private, private_sha = _json(args.private_manifest)
    private_by_video = _private_video_rows(private)
    archives_manifest = private.get("archives")
    if not isinstance(archives_manifest, dict):
        raise ValueError("private archive inventory is missing")
    model_sha = _sha256_file(args.mediapipe_model)
    dependency_sha = _sha256_file(args.dependency_lock)
    manual_values: list[np.ndarray] = []
    mediapipe_values: list[np.ndarray] = []
    detected: list[bool] = []
    participants: list[str] = []
    recordings: list[str] = []
    cohorts: list[str] = []
    tasks: list[str] = []
    video_digests: dict[str, str] = {}
    landmark_digests: dict[str, str] = {}
    extractor = MediaPipeFeatureExtractor(
        model_path=args.mediapipe_model,
        landmark_features="clinical23",
        capture_mirrored=None,
    )
    try:
        observed_videos: set[str] = set()
        for cohort in ("als", "healthy_control", "post_stroke"):
            video_id, video_relative, landmark_id, landmark_relative = ARCHIVES[cohort]
            video_path = args.data_root / video_relative
            landmark_path = args.data_root / landmark_relative
            video_expected = str(archives_manifest[video_id]["sha256"])
            landmark_expected = str(archives_manifest[landmark_id]["sha256"])
            with _open_pinned_zip(video_path, video_expected) as videos, _open_pinned_zip(
                landmark_path, landmark_expected
            ) as landmarks:
                video_digests[video_id] = video_expected
                landmark_digests[landmark_id] = landmark_expected
                video_members = _members_by_stem(videos, ".avi")
                landmark_members = _members_by_stem(landmarks, ".txt")
                if set(video_members) != set(landmark_members):
                    raise ValueError("video and manual landmark archive stems differ")
                for stem in sorted(video_members):
                    video_payload = videos.read(video_members[stem])
                    video_sha = hashlib.sha256(video_payload).hexdigest()
                    source = private_by_video.get(video_sha)
                    landmark_payload = landmarks.read(landmark_members[stem])
                    if (
                        source is None or source.get("cohort") != cohort
                        or source.get("video_archive_id") != video_id
                        or source.get("landmark_archive_id") != landmark_id
                        or hashlib.sha256(landmark_payload).hexdigest()
                        != source.get("landmark_sha256")
                    ):
                        raise ValueError("manual68 source member is not bound to private provenance")
                    observed_videos.add(video_sha)
                    manual_frames = parse_landmark_text(landmark_payload)
                    if len(manual_frames) != int(source["annotated_frames"]):
                        raise ValueError("manual68 frame count differs from private manifest")
                    temporary = _write_video(video_payload)
                    capture = cv2.VideoCapture(str(temporary))
                    try:
                        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                        if not capture.isOpened() or frame_count <= max(manual_frames):
                            raise ValueError("annotated frame lies outside its authenticated video")
                        for frame_index, points in sorted(manual_frames.items()):
                            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                                raise ValueError("decoder could not seek to an annotated frame")
                            ok, frame = capture.read()
                            landed = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                            if not ok or frame is None or landed != frame_index:
                                raise ValueError("decoder did not return the requested annotated frame")
                            manual_semantic = openface68_to_semantic23(points)
                            features, _nuisance = extractor.extract_frame_with_nuisance(frame)
                            present = features is not None
                            manual_values.append(manual_semantic)
                            if present:
                                array = np.asarray(features, dtype=np.float32)
                                if array.shape != (95,):
                                    raise ValueError("MediaPipe clinical23 feature schema drifted")
                                mediapipe_values.append(
                                    clinical23_v2_to_semantic23(array[-23:])
                                )
                            else:
                                mediapipe_values.append(np.full(23, np.nan, dtype=np.float32))
                            detected.append(present)
                            participants.append(str(source["participant_id"]))
                            recordings.append(str(source["recording_id"]))
                            cohorts.append(cohort)
                            tasks.append(str(source["task"]))
                    finally:
                        capture.release()
                        temporary.unlink(missing_ok=True)
        if observed_videos != set(private_by_video):
            raise ValueError("manual68 audit did not cover every private video")
    finally:
        extractor.close()
    if len(manual_values) != 3306:
        raise ValueError("manual68 audit did not cover all 3,306 annotated frames")
    report = build_manual68_audit_report(
        np.stack(manual_values), np.stack(mediapipe_values), np.asarray(detected),
        participant_ids=np.asarray(participants, dtype=object),
        recording_ids=np.asarray(recordings, dtype=object),
        cohorts=np.asarray(cohorts, dtype=object),
        tasks=np.asarray(tasks, dtype=object),
        provenance={
            "private_manifest_sha256": private_sha,
            "manual_landmark_collection_sha256": _collection_sha256(landmark_digests),
            "video_collection_sha256": _collection_sha256(video_digests),
            "mediapipe_model_sha256": model_sha,
            "implementation_sha256": _implementation_sha256(),
            "dependency_lock_sha256": dependency_sha,
        },
        runtime={
            "host": "nebius-h200",
            "device": "mediapipe_xnnpack_cpu",
            "seconds": float(time.perf_counter() - started),
        },
    )
    _write_no_overwrite_json(output, report)
    print(json.dumps({
        "schema_version": "neuroface_manual68_audit_v1_receipt",
        "report_sha256": _sha256_file(output),
        "annotated_frames": report["counts"]["annotated_frames"],
        "detected_frames": report["counts"]["detected_frames"],
        "measurement_gate_passed": report["decision"]["measurement_gate_passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
