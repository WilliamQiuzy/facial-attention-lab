"""Audit raw Mayo MediaPipe trajectories and derive clinical23 dynamics.

This is a data-readiness audit, not a classifier evaluation. It streams each
``landmarks.csv`` frame group, rejects malformed groups, computes the 23-feature
clinical geometry block, and summarizes rest-relative dynamics without loading
the multi-gigabyte CSV collection into memory at once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clinical_landmarks import (  # noqa: E402
    CLINICAL_LANDMARK_NAMES,
    CLINICAL_SIDE_CONVENTION,
    clinical_landmark_features,
)

_TOPOLOGY_FEATURE_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "fissure_h_mesh33_vs_mesh263",
        "fissure_h_mesh33",
        "fissure_h_mesh263",
    ),
    (
        "fissure_w_mesh33_vs_mesh263",
        "fissure_w_mesh33",
        "fissure_w_mesh263",
    ),
    (
        "eye_area_mesh33_vs_mesh263",
        "eye_area_mesh33",
        "eye_area_mesh263",
    ),
    (
        "brow_h_mesh33_vs_mesh263",
        "brow_h_mesh33",
        "brow_h_mesh263",
    ),
    (
        "corner_y_mesh61_vs_mesh291",
        "corner_y_mesh61",
        "corner_y_mesh291",
    ),
    (
        "corner_x_mesh61_vs_mesh291",
        "corner_x_mesh61",
        "corner_x_mesh291",
    ),
)


def _frame_groups(path: Path) -> Iterator[tuple[int, np.ndarray | None]]:
    """Yield ``(frame_id, points_or_none)``; malformed groups are explicit None."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "point_idx", "x", "y", "z"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        current: int | None = None
        points: dict[int, tuple[float, float, float]] = {}
        malformed = False

        def finish(frame_id: int, values: dict[int, tuple[float, float, float]], bad: bool):
            if bad or len(values) != 478 or set(values) != set(range(478)):
                return frame_id, None
            arr = np.asarray([values[i] for i in range(478)], dtype=np.float32)
            return frame_id, arr if np.isfinite(arr).all() else None

        for row in reader:
            try:
                frame_id = int(row["frame"])
                point_id = int(row["point_idx"])
                xyz = (float(row["x"]), float(row["y"]), float(row["z"]))
            except (TypeError, ValueError):
                if current is not None:
                    malformed = True
                continue
            if current is None:
                current = frame_id
            elif frame_id != current:
                yield finish(current, points, malformed)
                current, points, malformed = frame_id, {}, False
            if point_id in points:
                malformed = True
            points[point_id] = xyz
        if current is not None:
            yield finish(current, points, malformed)


def _finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite summary value: {value}")
    return value


def _feature_summary(
    values: np.ndarray,
    frame_ids: np.ndarray,
    expected_step: int,
) -> dict[str, float | int | str | None]:
    """Describe provisional whole-video motion without inventing action cues.

    Early valid frames are merely a reference, not a verified rest cue. AUC and
    velocity are computed only within contiguous sampled intervals, so missing
    detections never become an interpolated movement.
    """
    n = len(values)
    if expected_step < 1:
        raise ValueError("expected_step must be >= 1")
    baseline_n = max(1, min(10, int(math.ceil(n * 0.1))))
    early_reference = float(np.median(values[:baseline_n]))
    delta = values - early_reference
    peak_index = int(np.argmax(np.abs(delta)))
    peak = float(abs(delta[peak_index]))
    frame_delta = np.diff(frame_ids).astype(np.float64)
    if (frame_delta <= 0).any():
        raise ValueError("frame ids must be strictly increasing")
    contiguous = frame_delta == expected_step
    contiguous_interval_count = int(contiguous.sum())
    velocity = (np.diff(values)[contiguous] / frame_delta[contiguous]
                if contiguous_interval_count else None)
    auc = 0.0
    segment_start = 0
    for gap_i in np.flatnonzero(~contiguous):
        segment_end = int(gap_i) + 1
        if segment_end - segment_start > 1:
            auc += float(np.trapz(
                np.abs(delta[segment_start:segment_end]),
                x=frame_ids[segment_start:segment_end],
            ))
        segment_start = segment_end
    if n - segment_start > 1:
        auc += float(np.trapz(
            np.abs(delta[segment_start:n]), x=frame_ids[segment_start:n]))
    if n > 1:
        duration = float(frame_ids[-1] - frame_ids[0])
        peak_fraction = float(frame_ids[peak_index] - frame_ids[0]) / duration
    else:
        peak_fraction = 0.0
    last_to_peak = float(abs(delta[-1]) / (peak + 1e-8)) if peak > 0 else 0.0
    return {
        "median": _finite_float(np.median(values)),
        "p05": _finite_float(np.percentile(values, 5)),
        "p95": _finite_float(np.percentile(values, 95)),
        "robust_range": _finite_float(np.percentile(values, 95) - np.percentile(values, 5)),
        "temporal_std": _finite_float(np.std(values)),
        "early_reference": _finite_float(early_reference),
        "reference_semantics": "provisional_early_valid_frames_not_action_rest",
        "peak_abs_early_delta": _finite_float(peak),
        "contiguous_interval_count": contiguous_interval_count,
        "auc_abs_early_delta_contiguous": (
            _finite_float(auc) if contiguous_interval_count else None
        ),
        "max_abs_velocity_contiguous": (
            _finite_float(np.max(np.abs(velocity)))
            if velocity is not None else None
        ),
        "time_to_global_peak_fraction": _finite_float(peak_fraction),
        "last_valid_abs_early_delta_over_global_peak": _finite_float(last_to_peak),
    }


def _topology_pair_correlations(matrix: np.ndarray) -> dict[str, float | None]:
    name_to_i = {name: i for i, name in enumerate(CLINICAL_LANDMARK_NAMES)}
    required = {
        feature_name
        for _, first_name, second_name in _TOPOLOGY_FEATURE_PAIRS
        for feature_name in (first_name, second_name)
    }
    missing = sorted(required - set(name_to_i))
    if missing:
        raise ValueError(f"clinical topology-pair features missing: {missing}")
    output: dict[str, float | None] = {}
    for pair_name, first_name, second_name in _TOPOLOGY_FEATURE_PAIRS:
        first = matrix[:, name_to_i[first_name]]
        second = matrix[:, name_to_i[second_name]]
        if len(matrix) < 3:
            output[pair_name] = None
            continue
        if min(float(np.std(first)), float(np.std(second))) < 1e-8:
            output[pair_name] = None
        else:
            output[pair_name] = _finite_float(np.corrcoef(first, second)[0, 1])
    return output


def audit_landmark_csv(
    path: str | Path,
    image_width: int,
    image_height: int,
    stride: int = 1,
    sequence_output: str | Path | None = None,
    video_frame_count: int | None = None,
    source_video: str | Path | None = None,
) -> dict:
    """Audit a CSV against its video timeline and return serializable statistics."""
    path = Path(path)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if video_frame_count is None:
        raise ValueError("video_frame_count is required for coverage denominator")
    if video_frame_count <= 0:
        raise ValueError("video frame count must be positive")
    source_video_path: Path | None = None
    if sequence_output is not None:
        if source_video is None:
            raise ValueError("source_video is required for derived sequence provenance")
        source_video_path = Path(source_video).resolve()
        if not source_video_path.is_file():
            raise FileNotFoundError(f"source video does not exist: {source_video_path}")
        _, _, source_video_frame_count = _video_metadata(source_video_path)
        if source_video_frame_count != video_frame_count:
            raise ValueError(
                f"video_frame_count {video_frame_count} does not match "
                f"{source_video_path} metadata {source_video_frame_count}"
            )
    features: list[np.ndarray] = []
    frame_ids: list[int] = []
    total = valid = invalid = 0
    first_id: int | None = None
    last_id: int | None = None
    previous_id: int | None = None
    for frame_id, points in _frame_groups(path):
        if frame_id < 0 or frame_id >= video_frame_count:
            raise ValueError(
                f"{path}: frame id {frame_id} outside video timeline "
                f"[0, {video_frame_count - 1}]"
            )
        if previous_id is not None and frame_id <= previous_id:
            raise ValueError(f"{path}: frame ids must be strictly increasing")
        previous_id = frame_id
        total += 1
        first_id = frame_id if first_id is None else min(first_id, frame_id)
        last_id = frame_id if last_id is None else max(last_id, frame_id)
        if points is None:
            invalid += 1
            continue
        try:
            vector = clinical_landmark_features(points, image_width, image_height)
        except ValueError:
            invalid += 1
            continue
        valid += 1
        if frame_id % stride:
            continue
        features.append(vector)
        frame_ids.append(frame_id)
    if not features:
        raise ValueError(f"{path}: no valid landmark frames")
    matrix = np.stack(features)
    analyzed_frame_ids = np.asarray(frame_ids, dtype=np.int64)
    csv_sha256 = _sha256(path)
    timeline_frame_ids = np.arange(0, video_frame_count, stride, dtype=np.int64)
    timeline_matrix = np.full(
        (len(timeline_frame_ids), len(CLINICAL_LANDMARK_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    valid_mask = np.zeros(len(timeline_frame_ids), dtype=np.bool_)
    timeline_rows = analyzed_frame_ids // stride
    timeline_matrix[timeline_rows] = matrix.astype(np.float32, copy=False)
    valid_mask[timeline_rows] = True
    if total > video_frame_count:
        raise ValueError(
            f"{path}: {total} CSV frame groups exceed {video_frame_count} video frames"
        )
    feature_summaries = {
        name: _feature_summary(matrix[:, i], analyzed_frame_ids, expected_step=stride)
        for i, name in enumerate(CLINICAL_LANDMARK_NAMES)
    }
    topology_pair_correlations = _topology_pair_correlations(matrix)
    result = {
        "source": str(path),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "points_per_frame": 478,
        "feature_schema": "clinical23_v2",
        "side_convention": CLINICAL_SIDE_CONVENTION,
        "capture_mirrored": "unknown",
        "csv_sha256": csv_sha256,
        "feature_count": len(CLINICAL_LANDMARK_NAMES),
        "time_unit": "source_frame_id_not_seconds",
        "early_reference_assumption": (
            "median_first_10_percent_up_to_10_valid_frames_not_verified_action_rest"
        ),
        "video_frame_count": int(video_frame_count),
        "frames_total": total,
        "frames_valid_landmarks": valid,
        "frames_missing_from_csv": int(video_frame_count - total),
        "frames_analyzed": len(features),
        "frames_invalid": invalid,
        "first_frame_id": first_id,
        "last_frame_id": last_id,
        "csv_group_coverage": _finite_float(total / video_frame_count),
        "valid_landmark_coverage": _finite_float(valid / video_frame_count),
        "stride": stride,
        "features": feature_summaries,
        "topology_pair_trajectory_correlation": topology_pair_correlations,
    }
    if sequence_output is not None:
        sequence_output = Path(sequence_output)
        sequence_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{sequence_output.name}.", suffix=".npz",
            dir=sequence_output.parent, delete=False,
        ) as tmp:
            temporary_output = Path(tmp.name)
        try:
            np.savez_compressed(
                temporary_output,
                clinical23_seq=timeline_matrix,
                valid_mask=valid_mask,
                frame_ids=timeline_frame_ids,
                feature_schema=np.asarray("clinical23_v2"),
                feature_names=np.asarray(CLINICAL_LANDMARK_NAMES),
                side_convention=np.asarray(CLINICAL_SIDE_CONVENTION),
                capture_mirrored=np.asarray("unknown"),
                image_size=np.asarray((image_width, image_height), np.int32),
                stride=np.asarray(stride, np.int32),
                video_frame_count=np.asarray(video_frame_count, np.int64),
                timeline_kind=np.asarray("regular_source_frame_grid"),
                invalid_fill=np.asarray("nan"),
                source_csv=np.asarray(str(path.resolve())),
                source_csv_sha256=np.asarray(csv_sha256),
                source_video=np.asarray(str(source_video_path)),
            )
            os.replace(temporary_output, sequence_output)
        finally:
            temporary_output.unlink(missing_ok=True)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_video(csv_path: Path) -> Path:
    videos = [
        p for p in csv_path.parent.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".m4v"}
    ]
    candidates = [p for p in videos if "landmark" not in p.stem.lower()]
    if not candidates:
        # Existing Mayo exports retain only the rendered landmark overlay. It
        # has the same frame dimensions and timeline, so it can supply both
        # x/y aspect scale and the true coverage denominator.
        candidates = videos
    if not candidates:
        raise FileNotFoundError(
            f"{csv_path.parent}: no source video for dimensions/frame count"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"{csv_path.parent}: ambiguous source videos: "
            f"{[p.name for p in sorted(candidates)]}"
        )
    return candidates[0]


def _video_metadata(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open source video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid video dimensions for {path}: {width}x{height}")
    if frame_count <= 0:
        raise ValueError(f"invalid video frame count for {path}: {frame_count}")
    return width, height, frame_count


def _publish_derived_collection(staging_root: Path, derived_root: Path) -> None:
    """Replace a complete generated directory without ever mixing generations.

    The canonical directory is dedicated to generated trajectory NPZs. Unknown
    files or symlinks fail closed so a user-managed file is never deleted. If
    promotion fails after moving the old generation aside, the old directory is
    restored before the original error is re-raised.
    """
    backup_root: Path | None = None
    if derived_root.exists():
        if derived_root.is_symlink() or not derived_root.is_dir():
            raise RuntimeError(
                f"derived output must be a real directory: {derived_root}"
            )
        unexpected = [
            p.name for p in derived_root.iterdir()
            if not (p.is_file() and not p.is_symlink() and p.suffix == ".npz")
        ]
        if unexpected:
            raise RuntimeError(
                f"refusing to replace derived directory with unmanaged entries: "
                f"{sorted(unexpected)}"
            )
        backup_root = Path(tempfile.mkdtemp(
            prefix=f".{derived_root.name}.backup-", dir=derived_root.parent
        ))
        backup_root.rmdir()
        os.replace(derived_root, backup_root)

    try:
        os.replace(staging_root, derived_root)
    except Exception:
        if backup_root is not None:
            try:
                os.replace(backup_root, derived_root)
            except Exception as rollback_error:
                raise RuntimeError(
                    "derived collection promotion and rollback both failed; "
                    f"previous generation remains at {backup_root}"
                ) from rollback_error
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)


def audit_directory(
    input_root: Path,
    stride: int,
    derived_root: Path | None = None,
    allow_partial: bool = False,
) -> dict:
    if not input_root.is_dir():
        raise FileNotFoundError(f"audit input root does not exist: {input_root}")
    csv_paths = sorted(input_root.glob("*/landmarks.csv"))
    if not csv_paths:
        raise ValueError(f"audit input root contains no */landmarks.csv: {input_root}")
    staging_tmp: tempfile.TemporaryDirectory | None = None
    staging_root: Path | None = None
    if derived_root is not None:
        derived_root = Path(derived_root)
        derived_root.parent.mkdir(parents=True, exist_ok=True)
        staging_tmp = tempfile.TemporaryDirectory(
            prefix=f".{derived_root.name}.staging-", dir=derived_root.parent
        )
        staging_root = Path(staging_tmp.name)
    try:
        records = []
        failures = []
        hashes: dict[str, list[str]] = {}
        for csv_path in csv_paths:
            try:
                video = _source_video(csv_path)
                width, height, video_frame_count = _video_metadata(video)
                sequence_output = (staging_root / f"{csv_path.parent.name}.npz") \
                    if staging_root is not None else None
                record = audit_landmark_csv(
                    csv_path, width, height, stride=stride,
                    sequence_output=sequence_output,
                    video_frame_count=video_frame_count,
                    source_video=video,
                )
                digest = record["csv_sha256"]
                record.update({
                    "take": csv_path.parent.name,
                    "video": str(video),
                })
                records.append(record)
                hashes.setdefault(digest, []).append(csv_path.parent.name)
            except Exception as exc:  # noqa: BLE001 - preserve per-file failures
                failures.append({
                    "source": str(csv_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
        duplicates = [takes for takes in hashes.values() if len(takes) > 1]
        result = {
            "audit_kind": "mayo_clinical23_trajectory_readiness",
            "feature_normalization": "2d_similarity_roll_translation_scale",
            "classification_claim": "none_no_labels_or_controls",
            "input_root": str(input_root),
            "derived_root": str(derived_root) if derived_root is not None else None,
            "derived_publish_status": "not_requested" if derived_root is None else "staged",
            "derived_artifact_count": 0,
            "stride": stride,
            "record_count": len(records),
            "failure_count": len(failures),
            "duplicate_groups": duplicates,
            "records": records,
            "failures": failures,
        }
        if failures:
            if not allow_partial:
                raise RuntimeError(
                    f"audit failed for {len(failures)}/{len(csv_paths)} inputs; "
                    "rerun with --allow-partial only for an explicitly partial report"
                )
            if derived_root is not None:
                result["derived_publish_status"] = "not_published_partial_audit"
            return result

        if derived_root is not None:
            staged_outputs = sorted(staging_root.glob("*.npz"))
            if len(staged_outputs) != len(records):
                raise RuntimeError(
                    "complete audit did not produce exactly one staged sequence per record"
                )
            # Promote the whole directory as one generation. If the second
            # rename fails, the prior complete directory is rolled back; a
            # canonical directory can therefore never contain mixed runs.
            _publish_derived_collection(staging_root, derived_root)
            result["derived_publish_status"] = "published_complete_audit"
            result["derived_artifact_count"] = len(staged_outputs)
        return result
    finally:
        if staging_tmp is not None:
            staging_tmp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5,
                        help="analyze every Nth source frame; all groups are validated")
    parser.add_argument("--derived-root", type=Path,
                        help="optional directory for per-take clinical23 trajectory .npz files")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write a partial report despite per-take failures")
    args = parser.parse_args()
    result = audit_directory(
        args.input_root, stride=args.stride, derived_root=args.derived_root,
        allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "record_count", "failure_count", "duplicate_groups", "stride")}, indent=2))


if __name__ == "__main__":
    main()
