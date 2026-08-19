#!/usr/bin/env python3
"""Extract the locked Py-Feat XGBoost AU signal for the NeuroFace ALS study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.neuroface_au_v1 import (  # noqa: E402
    AU_NAMES,
    PAPER_AU_MODEL,
    PAPER_PYFEAT_VERSION,
    PAPER_TASKS,
    build_au_recording,
    load_au_recording_bytes,
    publish_au_cache,
    serialize_au_recording,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)


ENVIRONMENT_SCHEMA = "neuroface_pyfeat_xgb_environment_v1"
COLLECTION_SCHEMA = "neuroface_pyfeat_xgb_au_collection_v1"
FROZEN_BATCH_SIZE = 64
_HEX = set("0123456789abcdef")
_ALLOWED_ARCHIVE_IDS = {"als_videos", "healthy_control_videos"}
_EXTRACTION_TASK_PRIORITY = ("NSM_SPREAD", "NSM_KISS", "NSM_OPEN")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(document: object) -> bytes:
    return (json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("utf-8")


def _load_canonical_json(
    payload: bytes, *, identity: str, pretty: bool = False
) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 16 * 1024 * 1024:
        raise ValueError(f"{identity} must be exact bounded bytes")
    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{identity} must be strict UTF-8 JSON") from exc
    expected = (
        (json.dumps(
            document, sort_keys=True, indent=2, ensure_ascii=True,
            allow_nan=False,
        ) + "\n").encode("utf-8")
        if pretty else _canonical_json_bytes(document)
    )
    if not isinstance(document, dict) or expected != payload:
        raise ValueError(f"{identity} must be canonical JSON")
    return document


def _canonical_hex(value: object, *, name: str, prefix: str = "") -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{name} has invalid format")
    suffix = value[len(prefix):]
    if len(suffix) != 64 or any(character not in _HEX for character in suffix):
        raise ValueError(f"{name} has invalid digest")
    return value


def select_paper_records(manifest: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Select the exact complete ALS/healthy three-task Cartesian endpoint."""
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("records"), list):
        raise ValueError("private manifest must contain a records list")
    selected: list[dict[str, object]] = []
    seen_recordings: set[str] = set()
    seen_sources: set[str] = set()
    for raw in manifest["records"]:
        if not isinstance(raw, dict):
            raise ValueError("manifest record must be an object")
        cohort = raw.get("cohort")
        task = raw.get("task")
        if cohort not in {"als", "healthy_control"} or task not in PAPER_TASKS:
            continue
        row = dict(raw)
        recording_id = _canonical_hex(
            row.get("recording_id"), name="recording_id", prefix="rec_"
        )
        participant_id = _canonical_hex(
            row.get("participant_id"), name="participant_id", prefix="grp_"
        )
        source = _canonical_hex(row.get("video_sha256"), name="video_sha256")
        archive_id = row.get("video_archive_id")
        if archive_id not in _ALLOWED_ARCHIVE_IDS:
            raise ValueError("selected record has an unexpected video archive")
        if (isinstance(row.get("video_size_bytes"), bool)
                or not isinstance(row.get("video_size_bytes"), int)
                or row["video_size_bytes"] <= 0):
            raise ValueError("selected record has an invalid video size")
        expected_label = "affected" if cohort == "als" else "unaffected"
        if row.get("binary_label") != expected_label:
            raise ValueError("selected cohort and label disagree")
        if recording_id in seen_recordings or source in seen_sources:
            raise ValueError("selected recordings and source videos must be unique")
        seen_recordings.add(recording_id)
        seen_sources.add(source)
        row["recording_id"] = recording_id
        row["participant_id"] = participant_id
        row["video_sha256"] = source
        selected.append(row)

    grouped: dict[tuple[str, str], set[str]] = {}
    participant_cohorts: dict[str, str] = {}
    for row in selected:
        participant = str(row["participant_id"])
        cohort = str(row["cohort"])
        previous = participant_cohorts.setdefault(participant, cohort)
        if previous != cohort:
            raise ValueError("participant crosses cohorts")
        grouped.setdefault((cohort, participant), set()).add(str(row["task"]))
    cohort_counts = Counter(cohort for cohort, _ in grouped)
    if cohort_counts != Counter({"als": 11, "healthy_control": 11}):
        raise ValueError("paper endpoint must contain exactly 11 ALS and 11 healthy people")
    if any(tasks != set(PAPER_TASKS) for tasks in grouped.values()) or len(selected) != 66:
        raise ValueError("every paper participant must contain the exact three tasks")
    order = {task: index for index, task in enumerate(PAPER_TASKS)}
    return tuple(sorted(
        selected,
        key=lambda row: (
            str(row["cohort"]), str(row["participant_id"]), order[str(row["task"])]
        ),
    ))


def prioritize_extraction(
    selected: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Run the paper-comparable task first without changing endpoint membership."""
    if len(selected) != 66:
        raise ValueError("extraction priority requires the frozen 66 recordings")
    priority = {task: index for index, task in enumerate(_EXTRACTION_TASK_PRIORITY)}
    if any(row.get("task") not in priority for row in selected):
        raise ValueError("extraction priority received an unlocked task")
    output = tuple(dict(row) for row in sorted(
        selected,
        key=lambda row: (
            priority[str(row["task"])], str(row["cohort"]),
            str(row["participant_id"]),
        ),
    ))
    if {str(row["recording_id"]) for row in output} != {
        str(row["recording_id"]) for row in selected
    }:
        raise AssertionError("extraction priority changed endpoint membership")
    return output


def select_primary_faces(
    face_boxes: Sequence[Sequence[Sequence[float]]],
    au_predictions: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the largest detected face per frame and retain missingness."""
    if len(face_boxes) != len(au_predictions):
        raise ValueError("face boxes and AU predictions must have equal batch length")
    length = len(face_boxes)
    values = np.zeros((length, len(AU_NAMES)), dtype=np.float32)
    valid = np.zeros(length, dtype=bool)
    counts = np.zeros(length, dtype=np.int16)
    scores = np.zeros(length, dtype=np.float32)
    for index, (frame_boxes, frame_predictions) in enumerate(
        zip(face_boxes, au_predictions)
    ):
        boxes = list(frame_boxes)
        if len(boxes) > np.iinfo(np.int16).max:
            raise ValueError("face detector returned an implausible number of candidates")
        counts[index] = len(boxes)
        if not boxes:
            continue
        candidates = []
        for candidate_index, box in enumerate(boxes):
            if len(box) != 5:
                raise ValueError("face box must contain x1,y1,x2,y2,score")
            x1, y1, x2, y2, score = (float(value) for value in box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
                raise ValueError("face box must be finite")
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            candidates.append((area, score, -candidate_index, candidate_index))
        chosen = max(candidates)[-1]
        prediction = np.asarray(frame_predictions)
        if prediction.shape != (len(boxes), len(AU_NAMES)):
            raise ValueError("AU predictions must align one-to-one with detected faces")
        row = np.asarray(prediction[chosen], dtype=np.float32)
        score = float(boxes[chosen][4])
        if not np.isfinite(row).all() or not (0 < score <= 1):
            continue
        values[index] = row
        valid[index] = True
        scores[index] = np.float32(score)
    return values, valid, counts, scores


def validate_environment_lock(
    lock: Mapping[str, object],
    *,
    versions: Mapping[str, str],
    resource_payloads: Mapping[str, bytes],
    implementation_sha256: str,
) -> dict[str, object]:
    expected_keys = {
        "schema_version", "versions", "resources", "implementation_sha256"
    }
    if not isinstance(lock, Mapping) or set(lock) != expected_keys:
        raise ValueError("environment lock has an open or incomplete schema")
    if lock.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise ValueError("environment lock schema differs")
    frozen_versions = {
        "pyfeat": PAPER_PYFEAT_VERSION,
        "xgboost": "1.7.6",
        "torch": "2.2.1+cu121",
        "torchvision": "0.17.1+cu121",
    }
    if dict(lock.get("versions", {})) != frozen_versions or dict(versions) != frozen_versions:
        raise ValueError("runtime package versions differ from the freeze")
    expected_resources = lock.get("resources")
    if not isinstance(expected_resources, dict) or not expected_resources:
        raise ValueError("environment lock must bind model resources")
    if set(expected_resources) != set(resource_payloads):
        raise ValueError("runtime resource set differs from the freeze")
    for name, expected_digest in expected_resources.items():
        if (not isinstance(name, str) or PurePosixPath(name).name != name
                or not isinstance(expected_digest, str)
                or _canonical_hex(expected_digest, name="resource digest") != expected_digest
                or type(resource_payloads[name]) is not bytes
                or _digest(resource_payloads[name]) != expected_digest):
            raise ValueError(f"resource {name!r} differs from the freeze")
    _canonical_hex(implementation_sha256, name="implementation_sha256")
    if lock.get("implementation_sha256") != implementation_sha256:
        raise ValueError("extractor implementation differs from the freeze")
    return dict(lock)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src" / "datasets" / "neuroface_au_v1.py",
    ):
        payload = path.read_bytes()
        digest.update(path.name.encode("ascii") + b"\0")
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_zip_members(archive: zipfile.ZipFile):
    members = []
    seen = set()
    for info in archive.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        if (info.is_dir() or pure.is_absolute() or ".." in pure.parts
                or "\\" in name or name in seen or pure.suffix.lower() != ".avi"):
            if info.is_dir():
                continue
            raise ValueError("video archive contains an unsafe or unexpected member")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, stat.S_IFREG):
            raise ValueError("video archive member is not a regular file")
        seen.add(name)
        members.append(info)
    if not members:
        raise ValueError("video archive contains no AVI members")
    return tuple(members)


def _index_selected_members(
    archives: Mapping[str, Path],
    selected: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Path, str]]:
    wanted_by_archive: dict[str, set[str]] = {}
    for row in selected:
        wanted_by_archive.setdefault(str(row["video_archive_id"]), set()).add(
            str(row["video_sha256"])
        )
    result = {}
    for archive_id, wanted in wanted_by_archive.items():
        archive_path = archives.get(archive_id)
        if archive_path is None or not archive_path.is_file() or archive_path.is_symlink():
            raise ValueError(f"archive {archive_id!r} is unavailable")
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in _safe_zip_members(archive):
                digest = hashlib.sha256()
                with archive.open(info, "r") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                observed = digest.hexdigest()
                if observed in wanted:
                    if observed in result:
                        raise ValueError("selected video content occurs more than once")
                    result[observed] = (archive_path, info.filename)
        missing = wanted.difference(result)
        if missing:
            raise ValueError(f"archive {archive_id!r} is missing selected committed videos")
    return result


def _read_member_bytes(archive_path: Path, member_name: str, row: Mapping[str, object]) -> bytes:
    with zipfile.ZipFile(archive_path, "r") as archive:
        matching = [info for info in _safe_zip_members(archive) if info.filename == member_name]
        if len(matching) != 1:
            raise ValueError("selected archive member is no longer unique")
        payload = archive.read(matching[0])
    if len(payload) != row["video_size_bytes"] or _digest(payload) != row["video_sha256"]:
        raise ValueError("selected video bytes differ from the private manifest")
    return payload


def _build_au_only_detector(device: str):
    import torch
    from feat.detector import Detector
    from feat.pretrained import AU_LANDMARK_MAP, fetch_model
    from feat.utils import openface_2d_landmark_columns
    from feat.utils.io import get_resource_path

    detector = object.__new__(Detector)
    detector.device = torch.device(device)
    detector.verbose = False
    detector.info = {
        "face_model": "retinaface",
        "landmark_model": "mobilefacenet",
        "au_model": PAPER_AU_MODEL,
        "au_presence_columns": AU_LANDMARK_MAP["Feat"],
        "mapper": openface_2d_landmark_columns,
        "face_landmark_columns": openface_2d_landmark_columns,
    }
    detector.face_detector = fetch_model("face_model", "retinaface")(device=device)
    detector.landmark_detector = fetch_model("landmark_model", "mobilefacenet")(
        [112, 112], 136
    )
    checkpoint = torch.load(
        os.path.join(get_resource_path(), "mobilefacenet_model_best.pth.tar"),
        map_location=detector.device,
    )
    detector.landmark_detector.load_state_dict(checkpoint["state_dict"])
    detector.landmark_detector.eval()
    detector.au_model = fetch_model("au_model", PAPER_AU_MODEL)()
    if tuple(detector.info["au_presence_columns"]) != AU_NAMES:
        raise ValueError("Py-Feat AU order differs from the frozen schema")
    return detector


def _extract_video(detector, video_path: Path, row: Mapping[str, object]):
    import av
    import torch

    values_parts = []
    valid_parts = []
    count_parts = []
    score_parts = []
    decoded = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None:
            raise ValueError("video has no committed frame rate")
        fps = float(stream.average_rate)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video frame rate is invalid")
        batch = []
        for frame in container.decode(stream):
            batch.append(frame.to_ndarray(format="rgb24"))
            if len(batch) == FROZEN_BATCH_SIZE:
                _extract_batch(detector, batch, values_parts, valid_parts, count_parts, score_parts, torch)
                decoded += len(batch)
                batch = []
        if batch:
            _extract_batch(detector, batch, values_parts, valid_parts, count_parts, score_parts, torch)
            decoded += len(batch)
    if decoded <= 0:
        raise ValueError("video decoded no frames")
    recording = build_au_recording(
        recording_id=str(row["recording_id"]),
        group_id=str(row["participant_id"]),
        task=str(row["task"]),
        source_sha256=str(row["video_sha256"]),
        source_frame_count=decoded,
        fps=fps,
        frame_indices=np.arange(decoded, dtype=np.int64),
        timestamps=np.arange(decoded, dtype=np.float64) / fps,
        au_values=np.concatenate(values_parts),
        valid_mask=np.concatenate(valid_parts),
        selected_face_count=np.concatenate(count_parts),
        selected_face_score=np.concatenate(score_parts),
    )
    return recording


def _extract_batch(
    detector, batch, values_parts, valid_parts, count_parts, score_parts, torch_module
):
    images = torch_module.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
    with torch_module.inference_mode():
        boxes = detector.detect_faces(images)
        landmarks = detector.detect_landmarks(images, boxes)
    predictions = detector.detect_aus(images, landmarks)
    values, valid, counts, scores = select_primary_faces(boxes, predictions)
    values_parts.append(values)
    valid_parts.append(valid)
    count_parts.append(counts)
    score_parts.append(scores)


def _validate_cached(path: Path, row: Mapping[str, object]):
    recording = load_au_recording_bytes(path.read_bytes())
    expected = (
        row["recording_id"], row["participant_id"], row["task"], row["video_sha256"]
    )
    observed = (
        recording.recording_id, recording.group_id, recording.task,
        recording.source_sha256,
    )
    if observed != expected:
        raise ValueError("existing AU cache identity differs from the selected record")
    return recording


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--als-zip", type=Path, required=True)
    parser.add_argument("--healthy-zip", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_payload = args.private_manifest.read_bytes()
    if _digest(manifest_payload) != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("private manifest differs from the frozen NeuroFace inventory")
    manifest = _load_canonical_json(
        manifest_payload, identity="private manifest", pretty=True
    )
    selected = select_paper_records(manifest)
    extraction_order = prioritize_extraction(selected)

    lock_payload = args.environment_lock.read_bytes()
    lock = _load_canonical_json(lock_payload, identity="environment lock")
    import feat
    import torch
    import torchvision
    import xgboost
    from feat.utils.io import get_resource_path
    versions = {
        "pyfeat": feat.__version__,
        "xgboost": xgboost.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    resource_root = Path(get_resource_path())
    resource_payloads = {
        name: (resource_root / name).read_bytes()
        for name in lock.get("resources", {})
    }
    validate_environment_lock(
        lock,
        versions=versions,
        resource_payloads=resource_payloads,
        implementation_sha256=_implementation_digest(),
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise ValueError("locked extraction requires the verified NVIDIA H200")

    output_root = args.output_root
    if output_root.exists():
        if not output_root.is_dir() or output_root.is_symlink():
            raise ValueError("output root must be a real directory")
    else:
        output_root.mkdir(mode=0o700, parents=False)
    cache_root = output_root / "cache"
    cache_root.mkdir(mode=0o700, exist_ok=True)
    collection_path = output_root / "collection_manifest.json"
    if collection_path.exists() or collection_path.is_symlink():
        raise FileExistsError("collection manifest already exists")

    archives = {
        "als_videos": args.als_zip,
        "healthy_control_videos": args.healthy_zip,
    }
    member_index = _index_selected_members(archives, selected)
    detector = _build_au_only_detector("cuda")
    collection_rows = []
    for ordinal, row in enumerate(extraction_order, start=1):
        cache_path = cache_root / f"{row['recording_id']}.npz"
        if cache_path.exists():
            recording = _validate_cached(cache_path, row)
            cache_digest = _sha256_path(cache_path)
        else:
            archive_path, member_name = member_index[str(row["video_sha256"])]
            video_payload = _read_member_bytes(archive_path, member_name, row)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="neuroface-au-", suffix=".avi", dir=output_root
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o400)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    if handle.write(video_payload) != len(video_payload):
                        raise OSError("short temporary video write")
                    handle.flush()
                    os.fsync(handle.fileno())
                recording = _extract_video(detector, temporary, row)
            finally:
                try:
                    temporary.chmod(0o600)
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            cache_payload = serialize_au_recording(recording)
            cache_digest = publish_au_cache(cache_path, cache_payload)
        collection_rows.append({
            "ordinal": ordinal,
            "recording_id": recording.recording_id,
            "participant_id": recording.group_id,
            "cohort": row["cohort"],
            "task": recording.task,
            "source_sha256": recording.source_sha256,
            "cache_sha256": cache_digest,
            "frames": recording.source_frame_count,
            "valid_frames": int(recording.valid_mask.sum()),
            "coverage": recording.coverage,
        })
        print(json.dumps({
            "complete": ordinal,
            "total": len(selected),
            "task": recording.task,
            "coverage": recording.coverage,
        }, sort_keys=True), flush=True)

    collection = {
        "schema_version": COLLECTION_SCHEMA,
        "private_manifest_sha256": _digest(manifest_payload),
        "environment_lock_sha256": _digest(lock_payload),
        "implementation_sha256": _implementation_digest(),
        "counts": {
            "participants": 22,
            "recordings": 66,
            "als_participants": 11,
            "healthy_participants": 11,
            "tasks": {task: 22 for task in PAPER_TASKS},
        },
        "records": collection_rows,
    }
    publish_au_cache(collection_path, _canonical_json_bytes(collection))
    print(json.dumps(collection["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
