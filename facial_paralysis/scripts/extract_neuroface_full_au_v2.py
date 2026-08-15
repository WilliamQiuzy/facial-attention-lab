#!/usr/bin/env python3
"""Extract the locked Py-Feat AU signal for all 36 NeuroFace participants."""
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
from pathlib import Path
from pathlib import PurePosixPath
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_neuroface_au_v1 import (  # noqa: E402
    _build_au_only_detector,
    _canonical_hex,
    _canonical_json_bytes,
    _digest,
    _extract_batch,
    _load_canonical_json,
    _sha256_path,
    validate_environment_lock,
)
from src.datasets.neuroface_au_v2 import (  # noqa: E402
    build_full_au_recording,
    load_full_au_recording_bytes,
    publish_au_cache,
    serialize_full_au_recording,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)


COLLECTION_SCHEMA = "neuroface_pyfeat_xgb_sampled_au_collection_v2"
ENVIRONMENT_SCHEMA = "neuroface_pyfeat_xgb_environment_v2"
FROZEN_FRAME_STRIDE = 3
EXPECTED_RECORDINGS = 261
COHORT_PARTICIPANTS = {
    "als": 11,
    "healthy_control": 11,
    "post_stroke": 14,
}
COHORT_RECORDINGS = {
    "als": 76,
    "healthy_control": 80,
    "post_stroke": 105,
}
ALL_TASKS = (
    "NSM_SPREAD",
    "NSM_KISS",
    "NSM_OPEN",
    "NSM_BLOW",
    "NSM_BROW",
    "NSM_BIGSMILE",
    "DDK_PA",
    "DDK_PATAKA",
    "BBP_NORMAL",
)
PRIMARY_TASKS = frozenset(("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"))
EXPECTED_TASK_COUNTS = {
    "BBP_NORMAL": 34,
    "DDK_PA": 35,
    "DDK_PATAKA": 35,
    "NSM_BIGSMILE": 10,
    "NSM_BLOW": 24,
    "NSM_BROW": 15,
    "NSM_KISS": 36,
    "NSM_OPEN": 36,
    "NSM_SPREAD": 36,
}
ARCHIVE_BY_COHORT = {
    "als": "als_videos",
    "healthy_control": "healthy_control_videos",
    "post_stroke": "post_stroke_videos",
}
STROKE_METADATA = {
    "SLP_Assessment_PS.xlsx": (
        24675,
        "2f8d683d40f5b398528611a1dcd524841d06b17ece27ea4c60b8f61277611f4b",
    ),
    "VID_DATASET_Clinical information_Stroke.csv": (
        1109,
        "ccbcaa4b79d50c564523746879ad7293623a48fa788f0631781fd6db72de2661",
    ),
    "VideoInfoFile_PS.xlsx": (
        19877,
        "540990039fc4ed11c8f01c052c50f1ea7759dced265395033fe85a8d1c04a708",
    ),
}


def _safe_full_zip_members(
    archive: zipfile.ZipFile,
    archive_id: str,
) -> tuple[zipfile.ZipInfo, ...]:
    """Return only canonical videos while authenticating released sidecars."""
    if archive_id not in ARCHIVE_BY_COHORT.values():
        raise ValueError("archive identity differs from the frozen full cohort")
    videos: list[zipfile.ZipInfo] = []
    seen: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        mode = (info.external_attr >> 16) & 0o170000
        if (
            info.is_dir()
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or name in seen
            or mode not in (0, stat.S_IFREG)
            or bool(info.flag_bits & 0x1)
        ):
            raise ValueError("video archive contains an unsafe member")
        seen.add(name)
        if pure.suffix.lower() == ".avi":
            if (
                len(pure.parts) != 2
                or pure.parts[0] != "Videos"
                or info.file_size <= 0
                or info.file_size > 1024 * 1024 * 1024
                or info.compress_size <= 0
                or info.file_size > 200 * info.compress_size
            ):
                raise ValueError("video archive AVI member violates the closed layout")
            videos.append(info)
            continue
        commitment = STROKE_METADATA.get(name)
        if archive_id != "post_stroke_videos" or commitment is None:
            raise ValueError("video archive contains uncommitted non-video content")
        payload = archive.read(info)
        if len(payload) != commitment[0] or _digest(payload) != commitment[1]:
            raise ValueError("released stroke metadata differs from its commitment")
    if not videos:
        raise ValueError("video archive contains no canonical AVI members")
    return tuple(videos)


def _index_full_members(
    archives: Mapping[str, Path],
    selected: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Path, str, str]]:
    wanted_by_archive: dict[str, set[str]] = {}
    for row in selected:
        wanted_by_archive.setdefault(str(row["video_archive_id"]), set()).add(
            str(row["video_sha256"])
        )
    result: dict[str, tuple[Path, str, str]] = {}
    for archive_id, wanted in wanted_by_archive.items():
        archive_path = archives.get(archive_id)
        if archive_path is None or not archive_path.is_file() or archive_path.is_symlink():
            raise ValueError(f"archive {archive_id!r} is unavailable")
        found: set[str] = set()
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in _safe_full_zip_members(archive, archive_id):
                digest = hashlib.sha256()
                with archive.open(info, "r") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                observed = digest.hexdigest()
                if observed in wanted:
                    if observed in result:
                        raise ValueError("selected video content occurs more than once")
                    result[observed] = (archive_path, info.filename, archive_id)
                    found.add(observed)
        missing = wanted.difference(found)
        if missing:
            raise ValueError(f"archive {archive_id!r} is missing selected committed videos")
    return result


def _member_index_bytes(
    indexed: Mapping[str, tuple[Path, str, str]],
    selected: Sequence[Mapping[str, object]],
) -> bytes:
    expected = {str(row["video_sha256"]): row for row in selected}
    if len(expected) != EXPECTED_RECORDINGS or set(indexed) != set(expected):
        raise ValueError("member index coverage differs from the frozen endpoint")
    records = []
    for source in sorted(expected):
        value = indexed[source]
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("member index value is malformed")
        _, member_name, archive_id = value
        if archive_id != expected[source]["video_archive_id"]:
            raise ValueError("member index archive differs from private identity")
        pure = PurePosixPath(member_name)
        if (
            not isinstance(member_name, str)
            or pure.is_absolute() or ".." in pure.parts or "\\" in member_name
            or len(pure.parts) != 2 or pure.parts[0] != "Videos"
            or pure.suffix.lower() != ".avi"
        ):
            raise ValueError("member index contains a noncanonical video member")
        records.append({
            "source_sha256": source,
            "archive_id": archive_id,
            "member_name": member_name,
        })
    return _canonical_json_bytes({
        "schema_version": "neuroface_full_au_member_index_v1",
        "records": records,
    })


def _load_member_index(
    payload: bytes,
    archives: Mapping[str, Path],
    selected: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Path, str, str]]:
    if type(payload) is not bytes or not payload or len(payload) > 1024 * 1024:
        raise ValueError("member index must be bounded exact bytes")
    document = _load_canonical_json(payload, identity="member index")
    if (
        set(document) != {"schema_version", "records"}
        or document.get("schema_version") != "neuroface_full_au_member_index_v1"
        or not isinstance(document.get("records"), list)
        or len(document["records"]) != EXPECTED_RECORDINGS
    ):
        raise ValueError("member index schema or count differs")
    expected = {str(row["video_sha256"]): row for row in selected}
    result: dict[str, tuple[Path, str, str]] = {}
    for raw in document["records"]:
        if not isinstance(raw, dict) or set(raw) != {
            "source_sha256", "archive_id", "member_name"
        }:
            raise ValueError("member index row schema differs")
        source = _canonical_hex(raw["source_sha256"], name="source_sha256")
        row = expected.get(source)
        archive_id = raw["archive_id"]
        member_name = raw["member_name"]
        if (
            row is None or archive_id != row["video_archive_id"]
            or archive_id not in archives or source in result
        ):
            raise ValueError("member index row differs from private identity")
        pure = PurePosixPath(member_name) if isinstance(member_name, str) else None
        if (
            pure is None or pure.is_absolute() or ".." in pure.parts
            or "\\" in member_name or len(pure.parts) != 2
            or pure.parts[0] != "Videos" or pure.suffix.lower() != ".avi"
        ):
            raise ValueError("member index row contains a noncanonical path")
        result[source] = (archives[str(archive_id)], member_name, str(archive_id))
    if set(result) != set(expected):
        raise ValueError("member index omits committed source content")
    return result


def _read_full_member_bytes(
    archive_path: Path,
    member_name: str,
    archive_id: str,
    row: Mapping[str, object],
) -> bytes:
    with zipfile.ZipFile(archive_path, "r") as archive:
        matching = [
            info for info in _safe_full_zip_members(archive, archive_id)
            if info.filename == member_name
        ]
        if len(matching) != 1:
            raise ValueError("selected archive member is no longer unique")
        payload = archive.read(matching[0])
    if len(payload) != row["video_size_bytes"] or _digest(payload) != row["video_sha256"]:
        raise ValueError("selected video bytes differ from the private manifest")
    return payload


def _terminal_decode_is_complete(decoded: object, declared: object) -> bool:
    """Allow malformed trailer packets only after every declared frame decoded."""
    return (
        type(decoded) is int
        and type(declared) is int
        and declared > 0
        and decoded == declared
    )


def _extract_full_video(detector, video_path: Path, row: Mapping[str, object]):
    import av
    import torch

    values_parts = []
    valid_parts = []
    count_parts = []
    score_parts = []
    decoded = 0
    sampled_indices: list[int] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None:
            raise ValueError("video has no committed frame rate")
        fps = float(stream.average_rate)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video frame rate is invalid")
        declared_frames = int(stream.frames)
        batch = []
        try:
            for frame in container.decode(stream):
                source_index = decoded
                decoded += 1
                if source_index % FROZEN_FRAME_STRIDE != 0:
                    continue
                sampled_indices.append(source_index)
                batch.append(frame.to_ndarray(format="rgb24"))
                if len(batch) == 64:
                    _extract_batch(
                        detector, batch, values_parts, valid_parts,
                        count_parts, score_parts, torch,
                    )
                    batch = []
        except av.error.InvalidDataError:
            if not _terminal_decode_is_complete(decoded, declared_frames):
                raise
        if batch:
            _extract_batch(
                detector, batch, values_parts, valid_parts,
                count_parts, score_parts, torch,
            )
    if decoded <= 0:
        raise ValueError("video decoded no frames")
    return build_full_au_recording(
        recording_id=str(row["recording_id"]),
        group_id=str(row["participant_id"]),
        task=str(row["task"]),
        source_sha256=str(row["video_sha256"]),
        source_frame_count=decoded,
        fps=fps,
        sampling_stride=FROZEN_FRAME_STRIDE,
        frame_indices=np.asarray(sampled_indices, dtype=np.int64),
        timestamps=np.asarray(sampled_indices, dtype=np.float64) / fps,
        au_values=np.concatenate(values_parts),
        valid_mask=np.concatenate(valid_parts),
        selected_face_count=np.concatenate(count_parts),
        selected_face_score=np.concatenate(score_parts),
    )


def _validate_full_cached(path: Path, row: Mapping[str, object]):
    recording = load_full_au_recording_bytes(path.read_bytes())
    expected = (
        row["recording_id"], row["participant_id"], row["task"],
        row["video_sha256"],
    )
    observed = (
        recording.recording_id, recording.group_id, recording.task,
        recording.source_sha256,
    )
    if observed != expected:
        raise ValueError("existing full AU cache identity differs from selected record")
    return recording


def select_full_records(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Validate and select the exact released 36-person, 261-video cohort."""
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("records"), list):
        raise ValueError("private manifest must contain a records list")
    rows: list[dict[str, object]] = []
    seen_recordings: set[str] = set()
    seen_sources: set[str] = set()
    participant_cohort: dict[str, str] = {}
    participant_tasks: dict[str, set[str]] = {}
    for raw in manifest["records"]:
        if not isinstance(raw, dict):
            raise ValueError("manifest record must be an object")
        row = dict(raw)
        cohort = row.get("cohort")
        task = row.get("task")
        if cohort not in COHORT_PARTICIPANTS or task not in ALL_TASKS:
            raise ValueError("record cohort or task differs from the frozen full cohort")
        recording_id = _canonical_hex(
            row.get("recording_id"), name="recording_id", prefix="rec_"
        )
        participant_id = _canonical_hex(
            row.get("participant_id"), name="participant_id", prefix="grp_"
        )
        source = _canonical_hex(row.get("video_sha256"), name="video_sha256")
        if row.get("video_archive_id") != ARCHIVE_BY_COHORT[cohort]:
            raise ValueError("record cohort and archive identity disagree")
        expected_label = "unaffected" if cohort == "healthy_control" else "affected"
        if row.get("binary_label") != expected_label:
            raise ValueError("record cohort and binary outcome disagree")
        size = row.get("video_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("record video size must be a positive integer")
        if recording_id in seen_recordings or source in seen_sources:
            raise ValueError("recording identities and source videos must be unique")
        seen_recordings.add(recording_id)
        seen_sources.add(source)
        previous = participant_cohort.setdefault(participant_id, str(cohort))
        if previous != cohort:
            raise ValueError("one participant cannot cross clinical cohorts")
        tasks = participant_tasks.setdefault(participant_id, set())
        if str(task) in tasks:
            raise ValueError("one participant cannot repeat a task recording")
        tasks.add(str(task))
        row.update({
            "recording_id": recording_id,
            "participant_id": participant_id,
            "video_sha256": source,
        })
        rows.append(row)

    if len(rows) != EXPECTED_RECORDINGS:
        raise ValueError("full NeuroFace endpoint must contain exactly 261 recordings")
    cohort_participants = Counter(participant_cohort.values())
    cohort_recordings = Counter(str(row["cohort"]) for row in rows)
    task_counts = Counter(str(row["task"]) for row in rows)
    if cohort_participants != Counter(COHORT_PARTICIPANTS):
        raise ValueError("full NeuroFace participant counts differ from the freeze")
    if cohort_recordings != Counter(COHORT_RECORDINGS):
        raise ValueError("full NeuroFace recording counts differ from the freeze")
    if task_counts != Counter(EXPECTED_TASK_COUNTS):
        raise ValueError("full NeuroFace task counts differ from the freeze")
    if any(not PRIMARY_TASKS.issubset(tasks) for tasks in participant_tasks.values()):
        raise ValueError("every participant requires the three common primary actions")
    priority = {task: index for index, task in enumerate(ALL_TASKS)}
    return tuple(sorted(
        rows,
        key=lambda row: (
            str(row["cohort"]), str(row["participant_id"]),
            priority[str(row["task"])],
        ),
    ))


def prioritize_full_extraction(
    selected: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Extract the strongest common action first without changing membership."""
    if len(selected) != EXPECTED_RECORDINGS:
        raise ValueError("extraction priority requires the frozen full cohort")
    priority = {task: index for index, task in enumerate(ALL_TASKS)}
    if any(row.get("task") not in priority for row in selected):
        raise ValueError("extraction priority received an uncommitted task")
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


def deterministic_shard(
    ordered: Sequence[Mapping[str, object]], *, count: int, index: int,
) -> tuple[dict[str, object], ...]:
    """Partition only by frozen ordinal; labels and features never choose a shard."""
    if (
        isinstance(count, bool) or not isinstance(count, int)
        or isinstance(index, bool) or not isinstance(index, int)
        or count < 1 or count > 8 or index < 0 or index >= count
        or len(ordered) != EXPECTED_RECORDINGS
    ):
        raise ValueError("execution shard must be one of at most eight frozen shards")
    return tuple(dict(ordered[position]) for position in range(index, len(ordered), count))


def collection_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Recompute closed aggregate counts from validated selected rows."""
    participants = {str(row["participant_id"]) for row in rows}
    by_cohort: dict[str, set[str]] = {
        cohort: set() for cohort in COHORT_PARTICIPANTS
    }
    for row in rows:
        cohort = str(row["cohort"])
        if cohort not in by_cohort:
            raise ValueError("collection row contains an uncommitted cohort")
        by_cohort[cohort].add(str(row["participant_id"]))
    return {
        "participants": len(participants),
        "recordings": len(rows),
        "cohort_participants": {
            cohort: len(by_cohort[cohort]) for cohort in COHORT_PARTICIPANTS
        },
        "cohort_recordings": dict(Counter(str(row["cohort"]) for row in rows)),
        "tasks": dict(Counter(str(row["task"]) for row in rows)),
    }


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts" / "extract_neuroface_au_v1.py",
        ROOT / "src" / "datasets" / "neuroface_au_v1.py",
        ROOT / "src" / "datasets" / "neuroface_au_v2.py",
    ):
        payload = path.read_bytes()
        logical = path.relative_to(ROOT).as_posix().encode("ascii")
        digest.update(logical + b"\0" + len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--als-zip", type=Path, required=True)
    parser.add_argument("--healthy-zip", type=Path, required=True)
    parser.add_argument("--stroke-zip", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--member-index", type=Path)
    parser.add_argument("--build-member-index", type=Path)
    return parser


def _collection_row(
    ordinal: int, row: Mapping[str, object], recording, cache_digest: str,
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "recording_id": recording.recording_id,
        "participant_id": recording.group_id,
        "cohort": row["cohort"],
        "binary_label": row["binary_label"],
        "task": recording.task,
        "source_sha256": recording.source_sha256,
        "cache_sha256": cache_digest,
        "frames": recording.source_frame_count,
        "processed_frames": int(recording.frame_indices.size),
        "sampling_stride": recording.sampling_stride,
        "valid_frames": int(recording.valid_mask.sum()),
        "coverage": recording.coverage,
    }


def main() -> int:
    args = _parser().parse_args()
    manifest_payload = args.private_manifest.read_bytes()
    if _digest(manifest_payload) != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("private manifest differs from the frozen NeuroFace inventory")
    manifest = _load_canonical_json(
        manifest_payload, identity="private manifest", pretty=True
    )
    selected = select_full_records(manifest)
    extraction_order = prioritize_full_extraction(selected)
    if args.finalize_only and (args.shard_count != 1 or args.shard_index != 0):
        raise ValueError("finalization cannot be combined with execution sharding")
    if args.finalize_only and (args.member_index is not None or args.build_member_index is not None):
        raise ValueError("finalization does not consume or build an archive index")
    if args.build_member_index is not None and (
        args.member_index is not None or args.shard_count != 1 or args.shard_index != 0
    ):
        raise ValueError("member-index construction is one unsharded operation")
    if not args.finalize_only:
        execution_rows = deterministic_shard(
            extraction_order, count=args.shard_count, index=args.shard_index
        )

    lock_payload = args.environment_lock.read_bytes()
    lock = _load_canonical_json(lock_payload, identity="environment lock")
    if lock.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise ValueError("environment lock is not the full-cohort v2 contract")
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
    compatible_lock = dict(lock)
    compatible_lock["schema_version"] = "neuroface_pyfeat_xgb_environment_v1"
    validate_environment_lock(
        compatible_lock,
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

    if not args.finalize_only:
        archives = {
            "als_videos": args.als_zip,
            "healthy_control_videos": args.healthy_zip,
            "post_stroke_videos": args.stroke_zip,
        }
        if args.build_member_index is not None:
            indexed = _index_full_members(archives, selected)
            payload = _member_index_bytes(indexed, selected)
            digest = publish_au_cache(args.build_member_index, payload)
            print(json.dumps({
                "member_index_sha256": digest,
                "records": len(indexed),
            }, sort_keys=True))
            return 0
        if args.member_index is None:
            member_index = _index_full_members(archives, execution_rows)
        else:
            complete_index = _load_member_index(
                args.member_index.read_bytes(), archives, selected
            )
            member_index = {
                str(row["video_sha256"]): complete_index[str(row["video_sha256"])]
                for row in execution_rows
            }
        detector = _build_au_only_detector("cuda")
        ordinal_by_id = {
            str(row["recording_id"]): ordinal
            for ordinal, row in enumerate(extraction_order, start=1)
        }
        for shard_position, row in enumerate(execution_rows, start=1):
            cache_path = cache_root / f"{row['recording_id']}.npz"
            if cache_path.exists():
                recording = _validate_full_cached(cache_path, row)
            else:
                archive_path, member_name, archive_id = member_index[
                    str(row["video_sha256"])
                ]
                video_payload = _read_full_member_bytes(
                    archive_path, member_name, archive_id, row
                )
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f"neuroface-full-au-shard{args.shard_index}-",
                    suffix=".avi", dir=output_root,
                )
                temporary = Path(temporary_name)
                try:
                    os.fchmod(descriptor, 0o400)
                    with os.fdopen(descriptor, "wb", closefd=True) as handle:
                        if handle.write(video_payload) != len(video_payload):
                            raise OSError("short temporary video write")
                        handle.flush()
                        os.fsync(handle.fileno())
                    recording = _extract_full_video(detector, temporary, row)
                finally:
                    try:
                        temporary.chmod(0o600)
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                publish_au_cache(
                    cache_path, serialize_full_au_recording(recording)
                )
            print(json.dumps({
                "shard": args.shard_index,
                "shard_complete": shard_position,
                "shard_total": len(execution_rows),
                "ordinal": ordinal_by_id[recording.recording_id],
                "task": recording.task,
                "coverage": recording.coverage,
            }, sort_keys=True), flush=True)
        if args.shard_count != 1:
            print(json.dumps({
                "shard": args.shard_index, "complete": len(execution_rows)
            }, sort_keys=True))
            return 0

    expected_cache_paths = {
        cache_root / f"{row['recording_id']}.npz" for row in extraction_order
    }
    if set(cache_root.glob("*.npz")) != expected_cache_paths:
        raise ValueError("full AU cache set is incomplete or contains extras")
    collection_rows = []
    for ordinal, row in enumerate(extraction_order, start=1):
        cache_path = cache_root / f"{row['recording_id']}.npz"
        recording = _validate_full_cached(cache_path, row)
        collection_rows.append(_collection_row(
            ordinal, row, recording, _sha256_path(cache_path)
        ))

    collection = {
        "schema_version": COLLECTION_SCHEMA,
        "private_manifest_sha256": _digest(manifest_payload),
        "environment_lock_sha256": _digest(lock_payload),
        "implementation_sha256": _implementation_digest(),
        "counts": collection_counts(selected),
        "records": collection_rows,
    }
    publish_au_cache(collection_path, _canonical_json_bytes(collection))
    print(json.dumps(collection["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
