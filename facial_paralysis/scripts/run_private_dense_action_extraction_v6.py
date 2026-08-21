#!/usr/bin/env python3
"""H200-only dense action extraction from authenticated private manifests."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

from scripts.extract_dense_action_mesh_v1 import (
    build_dense_action_cache,
    extract_normalized_pair,
    publish_dense_action_cache,
    serialize_dense_action_cache,
)


MEEI_ACTIONS = (
    "BROW_RAISE",
    "EYE_GENTLE",
    "EYE_FORCEFUL",
    "SMILE_GENTLE",
    "SMILE_FULL",
    "LIP_PUCKER",
    "SHOW_BOTTOM_TEETH",
)
NEUROFACE_ACTIONS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
_LANDMARKER = None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_landmarker(model_path: str) -> None:
    global _LANDMARKER
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv

    _LANDMARKER = (
        mp,
        mpv.FaceLandmarker.create_from_options(
            mpv.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=model_path),
                running_mode=mpv.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
        ),
    )


def _detect_mesh(rgb: np.ndarray) -> np.ndarray | None:
    if _LANDMARKER is None:
        raise RuntimeError("worker FaceLandmarker is not initialized")
    mp, landmarker = _LANDMARKER
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if len(result.face_landmarks) != 1:
        return None
    points = result.face_landmarks[0]
    if len(points) != 478:
        return None
    return np.asarray([(p.x, p.y, p.z) for p in points], dtype=np.float64)


def _sample_interval(start: float, end: float, fps: float, count: int, total: int) -> np.ndarray:
    if not (0.0 <= start < end) or total < 1:
        raise ValueError("action interval is invalid")
    lo = max(0, min(total - 1, int(np.ceil(start * fps))))
    hi = max(lo, min(total - 1, int(np.floor(end * fps))))
    return np.rint(np.linspace(lo, hi, count)).astype(np.int64)


def _decode_selected(path: Path, indices: np.ndarray) -> tuple[dict[int, np.ndarray], int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot decode authenticated video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not np.isfinite(fps) or fps <= 0 or declared < 1:
        cap.release()
        raise ValueError("video clock is invalid")
    targets = set(int(value) for value in indices.reshape(-1).tolist())
    frames: dict[int, np.ndarray] = {}
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded in targets:
            frames[decoded] = frame
        decoded += 1
    cap.release()
    if decoded < 1 or max(targets) >= decoded:
        raise ValueError("sample grid falls outside exact decoded recording")
    return frames, decoded, fps


def _extract_grid(frames: dict[int, np.ndarray], indices: np.ndarray):
    shape = indices.shape + (478, 3)
    original = np.full(shape, np.nan, dtype=np.float64)
    mirrored = np.full(shape, np.nan, dtype=np.float64)
    valid = np.zeros(indices.shape, dtype=bool)
    for location, frame_index in np.ndenumerate(indices):
        first, second = extract_normalized_pair(frames[int(frame_index)], _detect_mesh)
        if first is None or second is None:
            continue
        original[location] = first
        mirrored[location] = second
        valid[location] = True
    return original, mirrored, valid


def _worker(task: dict[str, object]) -> dict[str, object]:
    path = Path(str(task["video_path"]))
    expected_sha = str(task["source_sha256"])
    if _sha256_file(path) != expected_sha:
        raise ValueError("worker video bytes differ from authenticated source")
    action_intervals = task["action_intervals"]
    baseline_interval = task["baseline_interval"]
    if not isinstance(action_intervals, list) or not isinstance(baseline_interval, list):
        raise ValueError("worker interval contract is invalid")

    probe = cv2.VideoCapture(str(path))
    if not probe.isOpened():
        raise ValueError("cannot probe authenticated video")
    fps = float(probe.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(probe.get(cv2.CAP_PROP_FRAME_COUNT)))
    probe.release()
    action_indices = np.stack(
        [
            _sample_interval(float(start), float(end), fps, int(task["action_samples"]), frame_count)
            for start, end in action_intervals
        ]
    )
    if len(baseline_interval) == 2:
        baseline_one = _sample_interval(
            float(baseline_interval[0]),
            float(baseline_interval[1]),
            fps,
            int(task["baseline_samples"]),
            frame_count,
        )
    elif len(baseline_interval) == 4:
        samples = int(task["baseline_samples"])
        if samples % 2:
            raise ValueError("two-edge baseline requires an even sample count")
        baseline_one = np.concatenate((
            _sample_interval(
                float(baseline_interval[0]), float(baseline_interval[1]),
                fps, samples // 2, frame_count,
            ),
            _sample_interval(
                float(baseline_interval[2]), float(baseline_interval[3]),
                fps, samples // 2, frame_count,
            ),
        ))
    else:
        raise ValueError("baseline interval contract is invalid")
    baseline_indices = np.repeat(baseline_one[None, :], len(action_intervals), axis=0)
    combined = np.concatenate((action_indices.reshape(-1), baseline_indices.reshape(-1)))
    frames, decoded_count, decoded_fps = _decode_selected(path, combined)
    if decoded_count != frame_count or abs(decoded_fps - fps) > 1e-9:
        raise ValueError("video probe and full decode disagree")
    original_actions, mirrored_actions, action_valid = _extract_grid(
        frames, action_indices
    )
    original_baselines, mirrored_baselines, baseline_valid = _extract_grid(
        frames, baseline_indices
    )
    cache = build_dense_action_cache(
        recording_id=str(task["recording_id"]),
        group_id=str(task["group_id"]),
        source_sha256=expected_sha,
        timing_sha256=str(task["timing_sha256"]),
        face_landmarker_sha256=str(task["face_landmarker_sha256"]),
        action_names=tuple(str(value) for value in task["action_names"]),
        source_frame_count=decoded_count,
        fps=fps,
        action_frame_indices=action_indices,
        baseline_frame_indices=baseline_indices,
        original_actions=original_actions,
        mirrored_actions=mirrored_actions,
        original_baselines=original_baselines,
        mirrored_baselines=mirrored_baselines,
        action_valid=action_valid,
        baseline_valid=baseline_valid,
    )
    payload = serialize_dense_action_cache(cache)
    output = Path(str(task["output_path"]))
    digest = publish_dense_action_cache(output, payload)
    return {
        "recording_id": cache.recording_id,
        "group_id": cache.group_id,
        "source_sha256": cache.source_sha256,
        "cache_sha256": digest,
        "action_valid": action_valid.sum(axis=1).astype(int).tolist(),
        "baseline_valid": baseline_valid.sum(axis=1).astype(int).tolist(),
    }


def _meei_tasks(args, private: dict, model_sha: str) -> list[dict[str, object]]:
    eligible = {
        str(row["source_sha256"]): row
        for row in private["media"]
        if row["media_type"] == "video" and row["dynamic_binary_eligible"] is True
    }
    paths: dict[str, Path] = {}
    for path in sorted(args.meei_root.rglob("*.mp4")):
        digest = _sha256_file(path)
        if digest in eligible:
            if digest in paths:
                raise ValueError("MEEI authenticated video digest is duplicated")
            paths[digest] = path
    if set(paths) != set(eligible):
        raise ValueError("MEEI raw root differs from its private manifest")
    tasks = []
    for digest, row in sorted(eligible.items()):
        timing_path = args.meei_intervals / f"{digest}.json"
        if not timing_path.is_file():
            continue
        timing_payload = timing_path.read_bytes()
        timing = json.loads(timing_payload)
        if timing["source_sha256"] != digest:
            raise ValueError("MEEI timing evidence source binding differs")
        intervals = {entry["action"]: entry for entry in timing["actions"]}
        if set(intervals) != {"REST", *MEEI_ACTIONS}:
            raise ValueError("MEEI action interval set differs")
        if any(
            not (
                0.0 <= float(intervals[name]["hold_start_seconds"])
                < float(intervals[name]["hold_end_seconds"])
            )
            for name in ("REST", *MEEI_ACTIONS)
        ):
            continue
        tasks.append(
            {
                "video_path": str(paths[digest]),
                "output_path": str(args.output_root / f"{row['recording_id']}.npz"),
                "recording_id": row["recording_id"],
                "group_id": row["participant_id"],
                "source_sha256": digest,
                "timing_sha256": _sha256_bytes(timing_payload),
                "face_landmarker_sha256": model_sha,
                "action_names": MEEI_ACTIONS,
                "action_intervals": [
                    [intervals[name]["hold_start_seconds"], intervals[name]["hold_end_seconds"]]
                    for name in MEEI_ACTIONS
                ],
                "baseline_interval": [
                    intervals["REST"]["hold_start_seconds"],
                    intervals["REST"]["hold_end_seconds"],
                ],
                "action_samples": 12,
                "baseline_samples": 6,
            }
        )
    if len(tasks) != 56:
        raise ValueError("MEEI authenticated timing subset must contain exact 56 recordings")
    return tasks


def _extract_neuroface_videos(args, selected: dict[str, dict]) -> dict[str, Path]:
    staging = args.output_root / "source-staging"
    staging.mkdir(mode=0o700)
    paths: dict[str, Path] = {}
    for archive_path in args.neuroface_zips:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.casefold().endswith(".avi"):
                    continue
                payload = archive.read(info)
                digest = _sha256_bytes(payload)
                if digest not in selected:
                    continue
                if digest in paths:
                    raise ValueError("NeuroFace authenticated video is duplicated")
                target = staging / f"{digest}.avi"
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    view = memoryview(payload)
                    written = 0
                    while written < len(payload):
                        count = os.write(descriptor, view[written:])
                        if count <= 0:
                            raise OSError("short write while staging NeuroFace video")
                        written += count
                    os.fsync(descriptor)
                except BaseException:
                    os.close(descriptor)
                    try:
                        target.unlink()
                    except OSError:
                        pass
                    raise
                else:
                    os.close(descriptor)
                paths[digest] = target
    if set(paths) != set(selected):
        raise ValueError("NeuroFace archives differ from the private manifest")
    return paths


def _neuroface_tasks(args, private: dict, manifest_sha: str, model_sha: str):
    selected = {
        str(row["video_sha256"]): row
        for row in private["records"]
        if str(row["task"]) in NEUROFACE_ACTIONS
    }
    if len(selected) != 108:
        raise ValueError("NeuroFace extraction requires exact 36x3 recordings")
    paths = _extract_neuroface_videos(args, selected)
    tasks = []
    for digest, row in sorted(selected.items()):
        probe = cv2.VideoCapture(str(paths[digest]))
        if not probe.isOpened():
            raise ValueError("cannot probe staged NeuroFace video")
        fps = float(probe.get(cv2.CAP_PROP_FPS))
        count = int(round(probe.get(cv2.CAP_PROP_FRAME_COUNT)))
        probe.release()
        duration = count / fps
        # Label-blind edge holds are the exogenous rest estimate; the central
        # 70% is the prompted action response for every cohort and label.
        edge = max(4.0 / fps, 0.15 * duration)
        tasks.append(
            {
                "video_path": str(paths[digest]),
                "output_path": str(args.output_root / f"{row['recording_id']}.npz"),
                "recording_id": row["recording_id"],
                "group_id": row["participant_id"],
                "source_sha256": digest,
                "timing_sha256": manifest_sha,
                "face_landmarker_sha256": model_sha,
                "action_names": (row["task"],),
                "action_intervals": [[edge, max(edge + 1.0 / fps, duration - edge)]],
                "baseline_interval": [
                    0.0,
                    min(edge, duration - 2.0 / fps),
                    max(edge, duration - edge),
                    duration - 1.0 / fps,
                ],
                "action_samples": 16,
                "baseline_samples": 6,
            }
        )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("meei", "neuroface"), required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--private-manifest-sha256", required=True)
    parser.add_argument("--face-landmarker", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--meei-root", type=Path)
    parser.add_argument("--meei-intervals", type=Path)
    parser.add_argument("--neuroface-zips", type=Path, nargs="*")
    args = parser.parse_args()
    manifest_payload = args.private_manifest.read_bytes()
    manifest_sha = _sha256_bytes(manifest_payload)
    if manifest_sha != args.private_manifest_sha256:
        raise ValueError("private manifest differs from its out-of-band pin")
    private = json.loads(manifest_payload)
    model_sha = _sha256_file(args.face_landmarker)
    if args.output_root.exists():
        if not args.output_root.is_dir() or any(args.output_root.iterdir()):
            raise FileExistsError("private output root must be an empty directory")
    else:
        args.output_root.mkdir(mode=0o700)
    if args.profile == "meei":
        if args.meei_root is None or args.meei_intervals is None:
            raise ValueError("MEEI roots are required")
        tasks = _meei_tasks(args, private, model_sha)
    else:
        if not args.neuroface_zips or len(args.neuroface_zips) != 3:
            raise ValueError("three NeuroFace archives are required")
        tasks = _neuroface_tasks(args, private, manifest_sha, model_sha)
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_landmarker,
        initargs=(str(args.face_landmarker),),
    ) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            print(json.dumps({"complete": len(results), "total": len(tasks)}), flush=True)
    collection = {
        "schema_version": "dense_action_mesh_collection_v1",
        "profile": args.profile,
        "private_manifest_sha256": manifest_sha,
        "face_landmarker_sha256": model_sha,
        "recordings": len(results),
        "records": sorted(results, key=lambda row: row["recording_id"]),
    }
    payload = (json.dumps(collection, sort_keys=True, separators=(",", ":")) + "\n").encode()
    publish_dense_action_cache(args.output_root / "collection_manifest.json", payload)
    staging = args.output_root / "source-staging"
    if staging.exists():
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
