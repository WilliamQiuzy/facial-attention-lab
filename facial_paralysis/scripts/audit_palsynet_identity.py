"""Create a local, deidentified identity-review audit for PalsyNet.

The script uses provenance-verified frozen MARLIN bundles only to rank possible
cross-recording identity matches.  Face contact sheets and the audit salt stay
under an ignored output directory.  The JSON manifest contains opaque IDs plus
per-record source digests needed for deidentified cache joins, never source
paths, filenames, or per-record cache paths.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import cv2
import numpy as np


EXPECTED_LABEL_COUNTS = {"affected": 27, "unaffected": 22}
MARLIN_WIDTH = 768
DEFAULT_TOP_PAIRS = 25
SEEK_POSITION_TOLERANCE_FRAMES = 0.25
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "palsynet_identity_audit"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID_RE = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID_RE = re.compile(r"^grp_[0-9a-f]{64}$")


@dataclass
class IdentityRecord:
    """One in-memory join; raw location/name fields are never serialized."""

    source_path: Path
    source_stem: str
    label: str
    source_sha256: str
    bundle_sha256: str
    recording_id: str
    group_id: str
    embedding: np.ndarray


@dataclass(frozen=True)
class BundleProvenanceRecord:
    """One verified local cache reference from a trusted extraction manifest."""

    bundle_path: Path
    bundle_sha256: str


def _require_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("source_sha256 must be 64 lowercase hexadecimal characters")
    return value


def _require_salt(salt: bytes) -> bytes:
    if not isinstance(salt, bytes) or len(salt) < 16:
        raise ValueError("audit salt must contain at least 128 bits")
    return salt


def _lexical_absolute(path: str | Path) -> Path:
    """Normalize ``.``/``..`` without following any filesystem symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _assert_no_symlink_components(
    path: str | Path,
    anchor: str | Path | None = None,
) -> Path:
    """Reject ``path`` links, plus descendants from an optional trusted anchor.

    Platform paths can legitimately have system-managed symlink ancestors (for
    example macOS ``/var``).  Security-sensitive descendant walks therefore
    name their real trust anchor explicitly instead of rejecting those global
    aliases.
    """
    absolute = _lexical_absolute(path)
    if anchor is None:
        if _is_symlink(absolute):
            raise ValueError("filesystem path component must not be a symlink")
        return absolute
    anchor_absolute = _lexical_absolute(anchor)
    relative = _relative_within(absolute, anchor_absolute)
    current = anchor_absolute
    if _is_symlink(current):
        raise ValueError("filesystem path component must not be a symlink")
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            # A child cannot exist below the first missing component.
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("filesystem path component must not be a symlink")
    return absolute


def _assert_tree_has_no_symlinks(root: str | Path) -> Path:
    """Reject a symlink root or any existing descendant without following it."""
    root = _assert_no_symlink_components(root)
    try:
        root_info = os.lstat(root)
    except FileNotFoundError:
        return root
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("audit output root must be a real directory")
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            candidate = Path(current) / name
            if stat.S_ISLNK(os.lstat(candidate).st_mode):
                raise ValueError("audit output tree must not contain symlinks")
    return root


def _relative_within(path: str | Path, root: str | Path) -> Path:
    path_absolute = _lexical_absolute(path)
    root_absolute = _lexical_absolute(root)
    try:
        return path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("path must stay within the trusted root") from exc


def _mkdirs_no_symlink(path: str | Path, root: str | Path) -> Path:
    """Create descendants one component at a time, rejecting link traversal."""
    root = _assert_tree_has_no_symlinks(root)
    relative = _relative_within(path, root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, mode=0o700)
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("audit output directory component is not a real directory")
    return current


def opaque_recording_id(source_sha256: str, salt: bytes) -> str:
    """Return the stable local recording pseudonym required by the audit."""
    digest = hmac.new(
        _require_salt(salt),
        _require_sha256(source_sha256).encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"rec_{digest}"


def default_group_id(source_sha256: str, salt: bytes) -> str:
    """Return an independent default one-recording group pseudonym."""
    digest = hmac.new(
        _require_salt(salt),
        b"group:" + _require_sha256(source_sha256).encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"grp_{digest}"


def _read_salt(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("audit salt must be a regular file")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("audit salt must not be group/world accessible")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            salt = handle.read()
    finally:
        os.close(fd)
    if len(salt) != 32:
        raise ValueError("audit salt must contain exactly 32 bytes")
    return salt


def load_or_create_salt(path: str | Path) -> bytes:
    """Securely create a 256-bit local salt, or reuse the existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        return _read_salt(path)
    except FileNotFoundError:
        pass

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    salt = secrets.token_bytes(32)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:  # another process won the creation race
        return _read_salt(path)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(salt)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600, follow_symlinks=False)
    except (NotImplementedError, OSError):
        pass
    return salt


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError("source video is missing or is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_marlin_embedding(path: str | Path) -> np.ndarray:
    """Load strict ``marlin (W,768)``, mean windows, and L2 normalize."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("matching frozen MARLIN bundle is missing")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if "marlin" not in bundle.files:
                raise ValueError("bundle is missing marlin")
            windows = np.asarray(bundle["marlin"])
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("cannot load a valid frozen MARLIN bundle") from exc
    if (
        windows.ndim != 2
        or windows.shape[0] < 1
        or windows.shape[1] != MARLIN_WIDTH
        or not np.issubdtype(windows.dtype, np.number)
        or not np.isrealobj(windows)
        or not np.isfinite(windows).all()
    ):
        raise ValueError("marlin must be a finite numeric array with shape (W, 768)")
    mean = windows.astype(np.float64, copy=False).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("mean MARLIN embedding must have a finite nonzero norm")
    return mean / norm


def _validated_bundle_key(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("bundle provenance key must be a nonempty relative path")
    key = PurePosixPath(value)
    if (
        key.is_absolute()
        or key.as_posix() != value
        or any(part in {"", ".", ".."} for part in key.parts)
        or key.name != "clip.npz"
    ):
        raise ValueError("bundle provenance key must be a normalized relative clip.npz path")
    return key


def load_bundle_provenance(
    path: str | Path,
    bundle_root: str | Path,
    expected_source_hashes: set[str],
) -> dict[str, BundleProvenanceRecord]:
    """Validate a trusted extraction-time source-to-bundle mapping.

    Existing legacy bundles do not contain their source digest.  Consequently
    this function never reconstructs provenance from current labels or stems:
    the caller must supply an independently generated extraction manifest.
    """
    if not isinstance(expected_source_hashes, set) or not expected_source_hashes:
        raise ValueError("bundle provenance requires a nonempty expected source set")
    for source_sha256 in expected_source_hashes:
        _require_sha256(source_sha256)
    provenance_path = _assert_no_symlink_components(path)
    if not provenance_path.is_file():
        raise ValueError(
            "trusted bundle provenance is missing; legacy bundles without embedded "
            "source hashes must be regenerated"
        )
    try:
        payload = json.loads(
            provenance_path.read_text(), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read valid trusted bundle provenance JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "dataset", "records"
    }:
        raise ValueError("bundle provenance has an unexpected top-level schema")
    if payload["schema_version"] != "palsynet_bundle_provenance_v1":
        raise ValueError("bundle provenance schema version is unsupported")
    if payload["dataset"] != "PalsyNet":
        raise ValueError("bundle provenance dataset must be PalsyNet")
    rows = payload["records"]
    if not isinstance(rows, list):
        raise ValueError("bundle provenance records must be a list")

    bundle_root = _assert_no_symlink_components(bundle_root)
    if not bundle_root.is_dir():
        raise ValueError("bundle root is missing or is not a real directory")
    result: dict[str, BundleProvenanceRecord] = {}
    seen_keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "source_sha256", "bundle_key", "bundle_sha256"
        }:
            raise ValueError("bundle provenance record has an unexpected schema")
        source_sha256 = _require_sha256(row["source_sha256"])
        bundle_sha256 = _require_sha256(row["bundle_sha256"])
        key = _validated_bundle_key(row["bundle_key"])
        key_text = key.as_posix()
        if source_sha256 in result:
            raise ValueError("bundle provenance repeats a source SHA-256")
        if key_text in seen_keys:
            raise ValueError("bundle provenance reuses one bundle key")
        seen_keys.add(key_text)
        bundle_path = _lexical_absolute(bundle_root.joinpath(*key.parts))
        _relative_within(bundle_path, bundle_root)
        _assert_no_symlink_components(bundle_path, bundle_root)
        if not bundle_path.is_file():
            raise ValueError("trusted provenance bundle is missing")
        observed_sha256 = sha256_file(bundle_path)
        if not hmac.compare_digest(observed_sha256, bundle_sha256):
            raise ValueError("trusted provenance bundle SHA-256 mismatch")
        result[source_sha256] = BundleProvenanceRecord(
            bundle_path=bundle_path,
            bundle_sha256=bundle_sha256,
        )
    if set(result) != expected_source_hashes:
        missing = expected_source_hashes - set(result)
        extra = set(result) - expected_source_hashes
        raise ValueError(
            f"bundle provenance coverage mismatch: {len(missing)} missing, "
            f"{len(extra)} extra"
        )
    return result


def collect_identity_records(
    video_root: str | Path,
    bundle_root: str | Path,
    salt: bytes,
    bundle_provenance: str | Path,
) -> list[IdentityRecord]:
    """Enumerate locked videos and join only provenance-verified bundles."""
    video_root, bundle_root = Path(video_root), Path(bundle_root)
    sources: list[tuple[str, Path]] = []
    for label, expected in EXPECTED_LABEL_COUNTS.items():
        label_root = video_root / label
        paths = sorted(
            (path for path in label_root.glob("*.mp4") if path.is_file()),
            key=lambda path: path.name,
        )
        if len(paths) != expected:
            raise ValueError(
                f"PalsyNet count mismatch for {label}: expected {expected}, "
                f"observed {len(paths)}"
            )
        sources.extend((label, path) for path in paths)
    expected_total = sum(EXPECTED_LABEL_COUNTS.values())
    if len(sources) != expected_total:
        raise ValueError(
            f"PalsyNet total mismatch: expected {expected_total}, observed {len(sources)}"
        )

    hashed_sources: list[tuple[str, Path, str]] = []
    seen_hashes: set[str] = set()
    for label, source_path in sources:
        source_sha256 = sha256_file(source_path)
        if source_sha256 in seen_hashes:
            raise ValueError("duplicate PalsyNet source SHA-256 detected")
        seen_hashes.add(source_sha256)
        hashed_sources.append((label, source_path, source_sha256))
    if len(seen_hashes) != expected_total:
        raise ValueError("PalsyNet source hashes are missing or duplicated")

    provenance = load_bundle_provenance(
        bundle_provenance, bundle_root, seen_hashes
    )
    records: list[IdentityRecord] = []
    for label, source_path, source_sha256 in hashed_sources:
        source_stem = source_path.stem
        bundle = provenance[source_sha256]
        embedding = load_marlin_embedding(bundle.bundle_path)
        records.append(IdentityRecord(
            source_path=source_path,
            source_stem=source_stem,
            label=label,
            source_sha256=source_sha256,
            bundle_sha256=bundle.bundle_sha256,
            recording_id=opaque_recording_id(source_sha256, salt),
            group_id=default_group_id(source_sha256, salt),
            embedding=embedding,
        ))
    return records


def rank_cosine_pairs(embeddings: Mapping[str, np.ndarray]) -> list[dict]:
    """Rank every distinct recording pair by cosine, with stable tie-breaking."""
    normalized: dict[str, np.ndarray] = {}
    width: int | None = None
    for recording_id in sorted(embeddings):
        if _RECORDING_ID_RE.fullmatch(recording_id) is None:
            raise ValueError("pair ranking requires canonical opaque recording ids")
        vector = np.asarray(embeddings[recording_id], dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("pair-ranking embeddings must be finite one-dimensional arrays")
        if width is None:
            width = int(vector.size)
        elif vector.size != width:
            raise ValueError("pair-ranking embeddings must share one width")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            raise ValueError("pair-ranking embeddings must have nonzero norm")
        normalized[recording_id] = vector / norm

    candidates: list[tuple[float, str, str]] = []
    for first, second in itertools.combinations(sorted(normalized), 2):
        cosine = float(np.dot(normalized[first], normalized[second]))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        candidates.append((cosine, first, second))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "rank": rank,
            "recording_id_a": first,
            "recording_id_b": second,
            "cosine": cosine,
        }
        for rank, (cosine, first, second) in enumerate(candidates, 1)
    ]


def validate_group_overrides(
    overrides: Mapping[str, str],
    records: Sequence[IdentityRecord],
) -> dict[str, str]:
    """Require one canonical reviewed group for every known recording."""
    if not isinstance(overrides, Mapping):
        raise ValueError("group override JSON must be an object mapping ids to ids")
    expected = {record.recording_id: record.label for record in records}
    if len(expected) != len(records):
        raise ValueError("recording ids must be unique before group review")
    actual = set(overrides)
    missing, unknown = set(expected) - actual, actual - set(expected)
    if missing or unknown:
        raise ValueError(
            f"group override coverage mismatch: {len(missing)} missing, "
            f"{len(unknown)} unknown"
        )

    validated: dict[str, str] = {}
    labels_by_group: dict[str, set[str]] = {}
    for recording_id, group_id in overrides.items():
        if _RECORDING_ID_RE.fullmatch(recording_id) is None:
            raise ValueError("group override contains a malformed recording id")
        if not isinstance(group_id, str) or _GROUP_ID_RE.fullmatch(group_id) is None:
            raise ValueError("group override contains a malformed group id")
        validated[recording_id] = group_id
        labels_by_group.setdefault(group_id, set()).add(expected[recording_id])
    if any(len(labels) != 1 for labels in labels_by_group.values()):
        raise ValueError("one reviewed identity group cannot cross labels")
    return validated


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in group override JSON")
        result[key] = value
    return result


def load_group_overrides(
    path: str | Path,
    records: Sequence[IdentityRecord],
) -> dict[str, str]:
    try:
        payload = json.loads(
            Path(path).read_text(), object_pairs_hook=_unique_json_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read valid group override JSON") from exc
    return validate_group_overrides(payload, records)


def _validate_records(records: Sequence[IdentityRecord]) -> list[IdentityRecord]:
    if not records:
        raise ValueError("identity manifest requires at least one recording")
    hashes: set[str] = set()
    recording_ids: set[str] = set()
    checked: list[IdentityRecord] = []
    for record in records:
        _require_sha256(record.source_sha256)
        _require_sha256(record.bundle_sha256)
        if record.source_sha256 in hashes:
            raise ValueError("duplicate source SHA-256 in identity manifest")
        hashes.add(record.source_sha256)
        if record.label not in EXPECTED_LABEL_COUNTS:
            raise ValueError("identity manifest label must be affected or unaffected")
        if _RECORDING_ID_RE.fullmatch(record.recording_id) is None:
            raise ValueError("identity manifest contains malformed recording id")
        if record.recording_id in recording_ids:
            raise ValueError("duplicate recording id in identity manifest")
        recording_ids.add(record.recording_id)
        if _GROUP_ID_RE.fullmatch(record.group_id) is None:
            raise ValueError("identity manifest contains malformed group id")
        embedding = np.asarray(record.embedding, dtype=np.float64)
        if embedding.shape != (MARLIN_WIDTH,) or not np.isfinite(embedding).all():
            raise ValueError("manifest embedding fingerprint requires finite MARLIN-768")
        embedding_norm = float(np.linalg.norm(embedding))
        if not math.isfinite(embedding_norm) or embedding_norm <= np.finfo(np.float64).eps:
            raise ValueError("manifest embedding fingerprint requires nonzero MARLIN-768")
        checked.append(record)
    return sorted(checked, key=lambda record: record.recording_id)


def _validate_pairs(pairs: Sequence[Mapping[str, object]], recording_ids: set[str]) -> list[dict]:
    expected_count = len(recording_ids) * (len(recording_ids) - 1) // 2
    if len(pairs) != expected_count:
        raise ValueError("ranked pair list must contain every cross-recording pair")
    seen: set[tuple[str, str]] = set()
    clean: list[dict] = []
    for expected_rank, pair in enumerate(pairs, 1):
        first = pair.get("recording_id_a")
        second = pair.get("recording_id_b")
        cosine = pair.get("cosine")
        if pair.get("rank") != expected_rank:
            raise ValueError("ranked pair list must have contiguous ranks")
        if first not in recording_ids or second not in recording_ids or first == second:
            raise ValueError("ranked pair contains an unknown or repeated recording")
        canonical = tuple(sorted((str(first), str(second))))
        if canonical in seen:
            raise ValueError("ranked pair list contains a duplicate pair")
        seen.add(canonical)
        try:
            cosine_value = float(cosine)
        except (TypeError, ValueError) as exc:
            raise ValueError("pair cosine must be numeric") from exc
        if not math.isfinite(cosine_value) or not -1.0 <= cosine_value <= 1.0:
            raise ValueError("pair cosine must be finite and within [-1, 1]")
        clean.append({
            "rank": expected_rank,
            "recording_id_a": str(first),
            "recording_id_b": str(second),
            "cosine": cosine_value,
        })
    return clean


def build_manifest(
    records: Sequence[IdentityRecord],
    ranked_pairs: Sequence[Mapping[str, object]],
    group_overrides: Mapping[str, str] | None = None,
    identity_review_status: str = "unreviewed",
    reviewer_evidence_sha256: str | None = None,
) -> dict:
    """Build the JSON-safe projection; raw record location/name is not copied."""
    checked = _validate_records(records)
    recording_ids = {record.recording_id for record in checked}
    pairs = _validate_pairs(ranked_pairs, recording_ids)
    groups = (
        validate_group_overrides(group_overrides, checked)
        if group_overrides is not None
        else {record.recording_id: record.group_id for record in checked}
    )
    if identity_review_status not in {"unreviewed", "reviewed"}:
        raise ValueError("identity review status must be unreviewed or reviewed")
    reviewed = identity_review_status == "reviewed"
    if reviewed:
        if group_overrides is None:
            raise ValueError("reviewed identity status requires complete group overrides")
        if (
            reviewer_evidence_sha256 is None
            or _SHA256_RE.fullmatch(reviewer_evidence_sha256) is None
        ):
            raise ValueError("reviewed identity status requires reviewer evidence SHA-256")
    elif reviewer_evidence_sha256 is not None:
        raise ValueError("reviewer evidence requires explicit reviewed identity status")
    review_status = identity_review_status
    claim_unit = "person_held_out" if reviewed else "video_held_out"

    source_fingerprint = hashlib.sha256()
    bundle_provenance_fingerprint = hashlib.sha256()
    embedding_fingerprint = hashlib.sha256()
    stable_records = sorted(
        checked, key=lambda item: (item.label, item.source_sha256)
    )
    for record in stable_records:
        source_fingerprint.update(
            f"{record.label}:{record.source_sha256}\n".encode("ascii")
        )
        bundle_provenance_fingerprint.update(
            f"{record.source_sha256}:{record.bundle_sha256}\n".encode("ascii")
        )
    for record in stable_records:
        embedding_fingerprint.update(
            f"{record.label}:{record.source_sha256}\n".encode("ascii")
        )
        embedding = np.array(record.embedding, dtype=np.float64, copy=True)
        embedding /= np.linalg.norm(embedding)
        embedding_fingerprint.update(
            np.asarray(embedding, dtype="<f8").tobytes(order="C")
        )

    counts = {
        label: sum(record.label == label for record in checked)
        for label in EXPECTED_LABEL_COUNTS
    }
    counts.update(total=len(checked), ranked_pairs=len(pairs))
    return {
        "schema_version": "palsynet_identity_audit_v1",
        "dataset": "PalsyNet",
        "claim_unit": claim_unit,
        "identity_review": {
            "status": review_status,
            "group_override_applied": group_overrides is not None,
            "manual_review_required": not reviewed,
            "reviewer_evidence_sha256": reviewer_evidence_sha256,
        },
        "counts": counts,
        "fingerprints": {
            "source_collection_sha256": source_fingerprint.hexdigest(),
            "bundle_provenance_sha256": bundle_provenance_fingerprint.hexdigest(),
            "embedding_collection_sha256": embedding_fingerprint.hexdigest(),
        },
        "contact_sheet_sampling": {
            "windows_per_video": 4,
            "window_size_frames": 32,
            "representative_frame_offset": 16,
            "raw_filename_text_burned_in": False,
        },
        "recordings": [
            {
                "recording_id": record.recording_id,
                "group_id": groups[record.recording_id],
                "label": record.label,
                "source_sha256": record.source_sha256,
                "identity_status": review_status,
                "claim_unit": claim_unit,
            }
            for record in checked
        ],
        "ranked_pairs": pairs,
    }


def deterministic_window_starts(
    frame_count: int,
    window_size: int = 32,
    n_windows: int = 4,
) -> tuple[int, ...]:
    """Return evenly spread, ordered, non-overlapping full-window starts."""
    if frame_count < 1 or window_size < 1 or n_windows < 1:
        raise ValueError("frame/window counts must be positive")
    required = window_size * n_windows
    if frame_count < required:
        raise ValueError(f"video requires at least {required} frames")
    if n_windows == 1:
        return ((frame_count - window_size) // 2,)
    last_start = frame_count - window_size
    starts = tuple((index * last_start) // (n_windows - 1)
                   for index in range(n_windows))
    if any(second - first < window_size
           for first, second in zip(starts, starts[1:])):
        raise ValueError("deterministic windows unexpectedly overlap")
    return starts


def read_representative_frames(path: str | Path) -> list[np.ndarray]:
    """Decode the center frame of each deterministic 32-frame audit window."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("identity contact-sheet video could not be opened")
    try:
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(raw_count) or raw_count < 1:
            raise RuntimeError("identity contact-sheet video has invalid frame metadata")
        frame_count = int(round(raw_count))
        centers = [start + 16 for start in deterministic_window_starts(frame_count)]
        frames: list[np.ndarray] = []
        for frame_index in centers:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise RuntimeError("identity contact-sheet seek failed")
            ok, frame = capture.read()
            if not ok or frame is None or frame.ndim != 3 or frame.size == 0:
                raise RuntimeError("identity contact-sheet frame decode failed")
            reported_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            expected_position = float(frame_index + 1)
            if (
                not math.isfinite(reported_position)
                or abs(reported_position - expected_position)
                > SEEK_POSITION_TOLERANCE_FRAMES
            ):
                raise RuntimeError(
                    "identity contact-sheet decoder landed on the wrong frame "
                    f"(expected post-read position {expected_position:.0f} within "
                    f"{SEEK_POSITION_TOLERANCE_FRAMES:.2f} frame)"
                )
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != 4:
        raise RuntimeError("identity contact sheet requires four decoded frames")
    return frames


def compose_contact_sheet(
    rows: Sequence[Sequence[np.ndarray]],
    panel_height: int = 240,
) -> np.ndarray:
    """Scale without distortion and pad panels; no identifying text is drawn."""
    if panel_height < 1 or not rows or any(not row for row in rows):
        raise ValueError("contact sheet requires nonempty rows and positive height")
    prepared: list[list[np.ndarray]] = []
    max_width = 0
    for row in rows:
        prepared_row: list[np.ndarray] = []
        for frame in row:
            image = np.asarray(frame)
            if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
                raise ValueError("contact-sheet frames must be nonempty BGR images")
            width = max(1, int(round(image.shape[1] * panel_height / image.shape[0])))
            interpolation = cv2.INTER_AREA if panel_height < image.shape[0] else cv2.INTER_LINEAR
            resized = cv2.resize(image, (width, panel_height), interpolation=interpolation)
            prepared_row.append(resized)
            max_width = max(max_width, width)
        prepared.append(prepared_row)

    rendered_rows: list[np.ndarray] = []
    for row in prepared:
        panels = [
            cv2.copyMakeBorder(
                image, 0, 0, 0, max_width - image.shape[1],
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            for image in row
        ]
        rendered_rows.append(np.hstack(panels))
    row_width = max(row.shape[1] for row in rendered_rows)
    rendered_rows = [
        cv2.copyMakeBorder(
            row, 0, 0, 0, row_width - row.shape[1],
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        for row in rendered_rows
    ]
    return np.vstack(rendered_rows)


def _write_image(path: Path, image: np.ndarray, trusted_root: Path) -> None:
    path = _lexical_absolute(path)
    trusted_root = _assert_tree_has_no_symlinks(trusted_root)
    _relative_within(path, trusted_root)
    encoded_ok, encoded = cv2.imencode(path.suffix, image)
    if not encoded_ok:
        raise RuntimeError("failed to encode local identity contact sheet")
    _write_bytes_exclusive(path, encoded.tobytes(), trusted_root, mode=0o600)


def generate_contact_sheets(
    records: Sequence[IdentityRecord],
    ranked_pairs: Sequence[Mapping[str, object]],
    output_root: str | Path,
    top_pairs: int,
) -> dict[str, int]:
    if top_pairs < 0:
        raise ValueError("top-pairs must be nonnegative")
    output_root = _assert_tree_has_no_symlinks(output_root)
    recording_root = output_root / "contact_sheets" / "recordings"
    pair_root = output_root / "contact_sheets" / "pairs"
    frames: dict[str, list[np.ndarray]] = {}
    for record in sorted(records, key=lambda item: item.recording_id):
        decoded = read_representative_frames(record.source_path)
        frames[record.recording_id] = decoded
        _write_image(
            recording_root / f"{record.recording_id}.jpg",
            compose_contact_sheet([decoded]),
            output_root,
        )
    selected = list(ranked_pairs[:top_pairs])
    for pair in selected:
        rank = int(pair["rank"])
        first, second = str(pair["recording_id_a"]), str(pair["recording_id_b"])
        _write_image(
            pair_root / f"pair_{rank:04d}.jpg",
            compose_contact_sheet([frames[first], frames[second]]),
            output_root,
        )
    return {"recordings": len(records), "ranked_pairs": len(selected)}


def validate_output_root(path: str | Path) -> Path:
    """Permit face sheets and salt only in the repository's ignored audit root."""
    absolute = _lexical_absolute(path)
    canonical = _lexical_absolute(CANONICAL_OUTPUT_ROOT)
    if absolute != canonical:
        raise ValueError(
            "output-root must be the canonical ignored "
            "facial_paralysis/outputs/palsynet_identity_audit directory"
        )
    return _assert_tree_has_no_symlinks(absolute)


def _write_bytes_exclusive(
    path: Path,
    payload: bytes,
    trusted_root: Path,
    mode: int = 0o600,
) -> None:
    path = _lexical_absolute(path)
    trusted_root = _assert_tree_has_no_symlinks(trusted_root)
    _relative_within(path, trusted_root)
    _mkdirs_no_symlink(path.parent, trusted_root)
    if path.exists() or _is_symlink(path):
        raise ValueError("audit generation destination must be fresh")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _write_manifest(
    path: Path,
    manifest: Mapping[str, object],
    trusted_root: Path,
) -> None:
    encoded = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    _write_bytes_exclusive(path, encoded, trusted_root)


def _read_regular_bytes(path: str | Path, description: str) -> bytes:
    path = _assert_no_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read {description}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{description} must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _try_relative_within(path: str | Path, root: str | Path) -> Path | None:
    try:
        return _relative_within(path, root)
    except ValueError:
        return None


def _paths_alias(first: str | Path, second: str | Path) -> bool:
    """Return whether two paths name the same file, including hard links."""
    first_path = _lexical_absolute(first)
    second_path = _lexical_absolute(second)
    if first_path == second_path:
        return True
    try:
        return os.path.samefile(first_path, second_path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError("cannot verify independent review evidence") from exc


def _validate_generation(
    staging_root: Path,
    manifest: Mapping[str, object],
    records: Sequence[IdentityRecord],
    ranked_pair_count: int,
    carried_files: set[Path],
) -> None:
    """Fail closed unless the fresh generation has exactly the declared files."""
    staging_root = _assert_tree_has_no_symlinks(staging_root)
    expected_files = {
        Path("identity_manifest.json"),
        *carried_files,
        *(Path("contact_sheets") / "recordings" / f"{record.recording_id}.jpg"
          for record in records),
        *(Path("contact_sheets") / "pairs" / f"pair_{rank:04d}.jpg"
          for rank in range(1, ranked_pair_count + 1)),
    }
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for current, directories, files in os.walk(staging_root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError("staged audit contains an unsafe directory")
            observed_directories.add(candidate.relative_to(staging_root))
        for filename in files:
            candidate = current_path / filename
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError("staged audit contains an unsafe file")
            if info.st_size < 1:
                raise ValueError("staged audit contains an empty file")
            observed_files.add(candidate.relative_to(staging_root))
    if observed_files != expected_files:
        raise ValueError("staged audit file set does not match the declared generation")

    expected_directories: set[Path] = set()
    for relative in expected_files:
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent)
            parent = parent.parent
    if observed_directories != expected_directories:
        raise ValueError("staged audit directory set contains stale or missing entries")

    manifest_path = staging_root / "identity_manifest.json"
    try:
        observed_manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("staged audit manifest cannot be decoded") from exc
    if observed_manifest != manifest:
        raise ValueError("staged audit manifest differs from the validated in-memory audit")
    contact = observed_manifest.get("contact_sheets")
    if not isinstance(contact, dict):
        raise ValueError("staged audit manifest is missing contact-sheet counts")
    if contact.get("recordings") != len(records):
        raise ValueError("staged recording contact-sheet count mismatch")
    if contact.get("ranked_pairs") != ranked_pair_count:
        raise ValueError("staged pair contact-sheet count mismatch")
    for relative in expected_files:
        if relative.suffix.lower() != ".jpg":
            continue
        image = cv2.imread(str(staging_root / relative), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.size == 0:
            raise ValueError("staged contact sheet is not a decodable image")


def _remove_generation_tree(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("generation cleanup target must be a real directory")
    shutil.rmtree(path)


def _promote_generation(staging_root: Path, output_root: Path) -> None:
    """Promote one complete directory and restore the prior audit on failure."""
    staging_root = _assert_tree_has_no_symlinks(staging_root)
    output_root = validate_output_root(output_root)
    parent = _assert_no_symlink_components(output_root.parent)
    backup = parent / f".{output_root.name}.backup-{secrets.token_hex(8)}"
    previous_moved = False
    try:
        if output_root.exists():
            os.replace(output_root, backup)
            previous_moved = True
        try:
            os.replace(staging_root, output_root)
        except BaseException:
            if previous_moved:
                if output_root.exists() or _is_symlink(output_root):
                    raise RuntimeError(
                        "cannot restore previous audit because output path reappeared"
                    )
                os.replace(backup, output_root)
                previous_moved = False
            raise
        if previous_moved:
            _remove_generation_tree(backup)
            previous_moved = False
    finally:
        if previous_moved and not output_root.exists() and backup.exists():
            os.replace(backup, output_root)


def run_audit(
    video_root: str | Path,
    bundle_root: str | Path,
    bundle_provenance: str | Path,
    output_root: str | Path,
    salt_file: str | Path | None = None,
    top_pairs: int = DEFAULT_TOP_PAIRS,
    group_overrides: str | Path | None = None,
    identity_review_status: str = "unreviewed",
    reviewer_evidence: str | Path | None = None,
) -> dict:
    output_root = validate_output_root(output_root)
    if top_pairs < 0:
        raise ValueError("top-pairs must be nonnegative")
    salt_path = _lexical_absolute(
        Path(salt_file) if salt_file is not None else output_root / "audit_salt.bin"
    )
    try:
        salt_relative = _relative_within(salt_path, output_root)
    except ValueError as exc:
        raise ValueError("salt-file must live under the ignored output-root") from exc
    _assert_no_symlink_components(salt_path)
    try:
        salt = _read_salt(salt_path)
    except FileNotFoundError:
        salt = secrets.token_bytes(32)

    if identity_review_status not in {"unreviewed", "reviewed"}:
        raise ValueError("identity review status must be unreviewed or reviewed")
    evidence_path: Path | None = None
    evidence_payload: bytes | None = None
    evidence_relative: Path | None = None
    if identity_review_status == "reviewed":
        if group_overrides is None or reviewer_evidence is None:
            raise ValueError(
                "reviewed identity status requires group overrides and reviewer evidence"
            )
        evidence_path = _lexical_absolute(reviewer_evidence)
        override_path = _lexical_absolute(group_overrides)
        if _paths_alias(evidence_path, override_path) or _paths_alias(
            evidence_path, salt_path
        ):
            raise ValueError(
                "reviewer evidence must be independent of group overrides and audit salt"
            )
        try:
            evidence_relative = _relative_within(evidence_path, output_root)
        except ValueError as exc:
            raise ValueError(
                "reviewer evidence must live under the ignored output-root"
            ) from exc
        evidence_payload = _read_regular_bytes(
            evidence_path, "reviewer evidence"
        )
        if not evidence_payload:
            raise ValueError("reviewer evidence must be a nonempty regular file")
        evidence_sha256 = hashlib.sha256(evidence_payload).hexdigest()
    else:
        if reviewer_evidence is not None:
            raise ValueError(
                "reviewer evidence requires --identity-review-status reviewed"
            )
        evidence_sha256 = None

    records = collect_identity_records(
        video_root, bundle_root, salt, bundle_provenance
    )
    pairs = rank_cosine_pairs({record.recording_id: record.embedding for record in records})
    overrides = (
        load_group_overrides(group_overrides, records)
        if group_overrides is not None else None
    )
    manifest = build_manifest(
        records,
        pairs,
        group_overrides=overrides,
        identity_review_status=identity_review_status,
        reviewer_evidence_sha256=evidence_sha256,
    )

    carried: dict[Path, bytes] = {salt_relative: salt}
    if evidence_path is not None and evidence_relative is not None:
        carried[evidence_relative] = evidence_payload or b""
    if group_overrides is not None:
        override_path = _lexical_absolute(group_overrides)
        override_relative = _try_relative_within(override_path, output_root)
        if override_relative is not None:
            carried[override_relative] = _read_regular_bytes(
                override_path, "group override JSON"
            )
    reserved = {Path("identity_manifest.json")}
    if reserved & set(carried):
        raise ValueError("carried local evidence conflicts with generated audit files")

    parent = _assert_no_symlink_components(output_root.parent)
    if not parent.is_dir():
        raise ValueError("canonical output parent must already be a real directory")
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging-",
        dir=parent,
    ))
    try:
        _assert_tree_has_no_symlinks(staging_root)
        for relative, payload in sorted(
            carried.items(), key=lambda item: item[0].as_posix()
        ):
            _write_bytes_exclusive(
                staging_root / relative,
                payload,
                staging_root,
                mode=0o600,
            )
        sheet_counts = generate_contact_sheets(
            records, pairs, staging_root, top_pairs
        )
        manifest["contact_sheets"] = {
            **sheet_counts,
            "storage": "local_ignored_output",
            "filenames": "opaque_ids_or_ranks_only",
        }
        _write_manifest(
            staging_root / "identity_manifest.json", manifest, staging_root
        )
        _validate_generation(
            staging_root=staging_root,
            manifest=manifest,
            records=records,
            ranked_pair_count=sheet_counts["ranked_pairs"],
            carried_files=set(carried),
        )
        _promote_generation(staging_root, output_root)
    finally:
        if staging_root.exists():
            _remove_generation_tree(staging_root)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument(
        "--bundle-provenance",
        required=True,
        type=Path,
        help=(
            "trusted extraction-time JSON mapping source SHA-256 to bundle key "
            "and bundle SHA-256; legacy stem-only caches are rejected"
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--salt-file", type=Path)
    parser.add_argument("--top-pairs", type=int, default=DEFAULT_TOP_PAIRS)
    parser.add_argument(
        "--group-overrides",
        type=Path,
        help=(
            "optional JSON object mapping every rec_ id to one grp_ id; "
            "mapping alone remains unreviewed"
        ),
    )
    parser.add_argument(
        "--identity-review-status",
        choices=("unreviewed", "reviewed"),
        default="unreviewed",
        help="reviewed requires group overrides plus separate reviewer evidence",
    )
    parser.add_argument(
        "--reviewer-evidence",
        type=Path,
        help="nonempty local evidence file under output-root; only its SHA-256 is stored",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_audit(
        video_root=args.video_root,
        bundle_root=args.bundle_root,
        bundle_provenance=args.bundle_provenance,
        output_root=args.output_root,
        salt_file=args.salt_file,
        top_pairs=args.top_pairs,
        group_overrides=args.group_overrides,
        identity_review_status=args.identity_review_status,
        reviewer_evidence=args.reviewer_evidence,
    )
    print(
        "PalsyNet identity audit complete: "
        f"{manifest['counts']['total']} recordings, "
        f"review status {manifest['identity_review']['status']}."
    )


if __name__ == "__main__":
    main()
