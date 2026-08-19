#!/usr/bin/env python3
"""Run the frozen exact-byte NeuroFace action-capacity experiment."""
from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.neuroface_action_capacity_v1 import (  # noqa: E402
    PRIMARY_TASKS,
    ActionCapacityAudit,
    ActionCapacityDataset,
    build_public_report,
    evaluate_action_capacity_oof,
    validate_public_report,
)
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
    mirror_action_capacity_features,
    neuroface_action_capacity_feature_vector,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    validate_neuroface_task_binding,
)
from scripts.launch_neuroface_action_capacity_v1 import (  # noqa: E402
    _directory_identity,
    _release_tree_commitment,
    _validate_frozen_input_release,
)


_MAX_PRIVATE_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_COLLECTION_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_CACHE_BYTES = 64 * 1024 * 1024
_MAX_DEPENDENCY_LOCK_BYTES = 32 * 1024 * 1024
_MAX_MOUNT_ATTESTATION_BYTES = 1024 * 1024
PINNED_DEPENDENCY_LOCK_SHA256 = (
    "f71f528af621e4eff83bb6a05c1fff09b0918ecda2cd9ac67979582b67767a6a"
)
PINNED_HOST_ATTESTATION_PUBLIC_KEY_DER_SHA256 = (
    "95229c6132a163e0ad073e5f6f8b9f3bdb8c7e52a292da3097310b24c1735904"
)
_HOST_ATTESTATION_PUBLIC_KEY_PATH = (
    PROJECT_ROOT / "environment" / "neuroface_action_capacity_host_audit_ed25519_public.pem"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_NEUROFACE_INPUT_ROOT = Path("/neuroface-input")
_NEUROFACE_OUTPUT_ROOT = Path("/neuroface-output")
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "evaluation" / "neuroface_action_capacity_v1.py",
    PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
    PROJECT_ROOT / "src" / "datasets" / "patient_multistream.py",
    PROJECT_ROOT / "src" / "preprocessing" / "action_capacity_features_v1.py",
    PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
    PROJECT_ROOT / "src" / "preprocessing" / "generalization_110d.py",
    PROJECT_ROOT / "src" / "preprocessing" / "script_action_segmentation_v1.py",
    PROJECT_ROOT / "src" / "preprocessing" / "trajectory_features.py",
    PROJECT_ROOT / "src" / "training" / "neuroface_motion_pretrain_v1.py",
    PROJECT_ROOT / "scripts" / "run_mirror_invariant_110d.py",
    PROJECT_ROOT / "scripts" / "launch_neuroface_action_capacity_v1.py",
    _HOST_ATTESTATION_PUBLIC_KEY_PATH,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", required=True, type=Path)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--mount-attestation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _verify_h200_runtime() -> None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("formal release requires host-visible NVIDIA GPU evidence") from exc
    names = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if not any(name in {"NVIDIA H200", "NVIDIA H200 NVL"} for name in names):
        raise ValueError("formal release requires an NVIDIA H200 host")


@dataclass
class _InputAccessAudit:
    authenticated_neuroface_files_read: int = 0
    prohibited_path_attempts: int = 0
    process_mount_table_checked: bool = False

    def audit_process_mount_table(self, attestation: "_VerifiedHostAttestation") -> None:
        mountinfo = Path("/proc/self/mountinfo")
        if not mountinfo.is_file():
            raise ValueError("formal H200 run requires Linux process mount evidence")
        payload = mountinfo.read_bytes()
        try:
            _validate_container_mountinfo(payload, attestation=attestation)
        except ValueError:
            self.prohibited_path_attempts += 1
            raise
        self.process_mount_table_checked = True

    def formal_action_audit(self) -> ActionCapacityAudit:
        if self.authenticated_neuroface_files_read != 112:
            raise ValueError(
                "formal run must authenticate 3 authorities, 108 caches, and mount evidence"
            )
        if self.prohibited_path_attempts != 0:
            raise ValueError("a prohibited path was attempted")
        if self.process_mount_table_checked is not True:
            raise ValueError("formal run lacks process mount-table evidence")
        return ActionCapacityAudit(
            palsynet_path_accesses=0,
            palsynet_cache_reads=0,
            palsynet_predictions=0,
        )


@dataclass(frozen=True)
class _VerifiedHostAttestation:
    sha256: str
    payload: dict[str, object]


def _reject_palsynet_path(path: Path, *, audit: _InputAccessAudit | None = None) -> None:
    """Reject protected-data-looking arguments without touching the path."""
    if "palsynet" in os.fspath(path).casefold():
        if audit is not None:
            audit.prohibited_path_attempts += 1
        raise ValueError("protected-data paths are outside this experiment")


def _reject_symlink_components(path: Path) -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("authenticated paths must be absolute")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise ValueError("authenticated path component does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("authenticated paths cannot traverse symlink components")


def _read_exact_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    audit: _InputAccessAudit | None = None,
) -> tuple[bytes, str]:
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer")
    _reject_palsynet_path(path, audit=audit)
    _reject_symlink_components(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("authenticated input must be an existing non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError("authenticated input is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) == 0 or len(payload) > maximum_bytes
            or (
                before.st_dev, before.st_ino, before.st_mode, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns,
            )
            != (
                after.st_dev, after.st_ino, after.st_mode, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            )
            or len(payload) != before.st_size
        ):
            raise ValueError("authenticated input changed or exceeded its bound")
        _reject_symlink_components(path)
        leaf = os.lstat(path)
        if (
            leaf.st_dev, leaf.st_ino, leaf.st_mode, leaf.st_nlink,
            leaf.st_size, leaf.st_mtime_ns, leaf.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
            after.st_size, after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ValueError("authenticated path no longer names the opened inode")
        if audit is not None:
            audit.authenticated_neuroface_files_read += 1
        return payload, hashlib.sha256(payload).hexdigest()
    finally:
        os.close(descriptor)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("authenticated JSON contains a duplicate key")
        result[key] = value
    return result


def _json_bytes(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authenticated input is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("authenticated JSON must contain an object")
    return value


def _canonical_signed_payload(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _public_key_der(pem: bytes) -> bytes:
    if type(pem) is not bytes:
        raise ValueError("host audit public key must be exact bytes")
    lines = pem.splitlines()
    if (
        len(lines) != 3
        or lines[0] != b"-----BEGIN PUBLIC KEY-----"
        or lines[2] != b"-----END PUBLIC KEY-----"
    ):
        raise ValueError("host audit public key PEM differs from the frozen format")
    try:
        der = base64.b64decode(lines[1], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("host audit public key PEM is malformed") from exc
    if hashlib.sha256(der).hexdigest() != PINNED_HOST_ATTESTATION_PUBLIC_KEY_DER_SHA256:
        raise ValueError("host audit public key differs from its DER fingerprint pin")
    return der


_ED25519_P = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493
_ED25519_D = (-121665 * pow(121666, _ED25519_P - 2, _ED25519_P)) % _ED25519_P
_ED25519_I = pow(2, (_ED25519_P - 1) // 4, _ED25519_P)
_ED25519_IDENTITY = (0, 1, 1, 0)


def _ed25519_decode_point(encoded: bytes) -> tuple[int, int, int, int]:
    if type(encoded) is not bytes or len(encoded) != 32:
        raise ValueError("Ed25519 compressed point must contain exactly 32 bytes")
    integer = int.from_bytes(encoded, "little")
    sign = integer >> 255
    y = integer & ((1 << 255) - 1)
    if y >= _ED25519_P:
        raise ValueError("Ed25519 point has a non-canonical y coordinate")
    y_squared = y * y % _ED25519_P
    x_squared = (
        (y_squared - 1)
        * pow((_ED25519_D * y_squared + 1) % _ED25519_P, _ED25519_P - 2, _ED25519_P)
    ) % _ED25519_P
    x = pow(x_squared, (_ED25519_P + 3) // 8, _ED25519_P)
    if x * x % _ED25519_P != x_squared:
        x = x * _ED25519_I % _ED25519_P
    if x * x % _ED25519_P != x_squared:
        raise ValueError("Ed25519 compressed point is not on the curve")
    if (x & 1) != sign:
        x = _ED25519_P - x
    if x == 0 and sign:
        raise ValueError("Ed25519 compressed point uses a non-canonical sign bit")
    return x, y, 1, x * y % _ED25519_P


def _ed25519_add(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _ED25519_P
    b = (y1 + x1) * (y2 + x2) % _ED25519_P
    c = 2 * _ED25519_D * t1 * t2 % _ED25519_P
    d = 2 * z1 * z2 % _ED25519_P
    e = (b - a) % _ED25519_P
    f = (d - c) % _ED25519_P
    g = (d + c) % _ED25519_P
    h = (b + a) % _ED25519_P
    return e * f % _ED25519_P, g * h % _ED25519_P, f * g % _ED25519_P, e * h % _ED25519_P


def _ed25519_scalar_multiply(
    scalar: int,
    point: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    result = _ED25519_IDENTITY
    addend = point
    value = scalar
    while value:
        if value & 1:
            result = _ed25519_add(result, addend)
        addend = _ed25519_add(addend, addend)
        value >>= 1
    return result


def _ed25519_equal(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return (
        left[0] * right[2] % _ED25519_P == right[0] * left[2] % _ED25519_P
        and left[1] * right[2] % _ED25519_P == right[1] * left[2] % _ED25519_P
    )


def _verify_ed25519_signature(message: bytes, signature: bytes) -> None:
    if type(message) is not bytes or not message or type(signature) is not bytes:
        raise ValueError("signed host audit message or signature is malformed")
    if len(signature) != 64:
        raise ValueError("Ed25519 host audit signature must contain exactly 64 bytes")
    public_pem, _ = _read_exact_regular_file(
        _HOST_ATTESTATION_PUBLIC_KEY_PATH, maximum_bytes=4096,
    )
    public_der = _public_key_der(public_pem)
    prefix = bytes.fromhex("302a300506032b6570032100")
    if len(public_der) != len(prefix) + 32 or not public_der.startswith(prefix):
        raise ValueError("host audit public key is not an Ed25519 SubjectPublicKeyInfo")
    public_encoded = public_der[len(prefix):]
    public_point = _ed25519_decode_point(public_encoded)
    if (
        _ed25519_equal(public_point, _ED25519_IDENTITY)
        or not _ed25519_equal(
            _ed25519_scalar_multiply(_ED25519_L, public_point),
            _ED25519_IDENTITY,
        )
    ):
        raise ValueError("host audit public key is not a prime-order Ed25519 point")
    r_encoded = signature[:32]
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _ED25519_L:
        raise ValueError("host audit Ed25519 signature scalar is non-canonical")
    r_point = _ed25519_decode_point(r_encoded)
    if _ed25519_equal(
        _ed25519_scalar_multiply(8, r_point), _ED25519_IDENTITY
    ):
        raise ValueError("host audit Ed25519 signature uses a small-order R point")
    base_y = 4 * pow(5, _ED25519_P - 2, _ED25519_P) % _ED25519_P
    base_point = _ed25519_decode_point(base_y.to_bytes(32, "little"))
    challenge = int.from_bytes(
        hashlib.sha512(r_encoded + public_encoded + message).digest(), "little"
    ) % _ED25519_L
    left = _ed25519_scalar_multiply(scalar, base_point)
    right = _ed25519_add(
        r_point, _ed25519_scalar_multiply(challenge, public_point)
    )
    if not _ed25519_equal(left, right):
        raise ValueError("host audit Ed25519 signature verification failed")


def _validate_mount_attestation(
    payload: bytes,
) -> _VerifiedHostAttestation:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > _MAX_MOUNT_ATTESTATION_BYTES
    ):
        raise ValueError("mount attestation bytes are malformed")
    observed = hashlib.sha256(payload).hexdigest()
    envelope = _json_bytes(payload)
    canonical = (
        json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if canonical != payload or set(envelope) != {
        "schema_version", "payload", "signature_base64",
    } or envelope.get("schema_version") != (
        "neuroface_action_capacity_signed_host_attestation_v1"
    ):
        raise ValueError("signed host attestation envelope differs from its closed schema")
    value = envelope.get("payload")
    signature_text = envelope.get("signature_base64")
    if not isinstance(value, dict) or not isinstance(signature_text, str):
        raise ValueError("signed host attestation payload or signature is malformed")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("signed host attestation signature is not strict base64") from exc
    signed_bytes = _canonical_signed_payload(value)
    _verify_ed25519_signature(signed_bytes, signature)
    input_source = (
        "/home/ssh-ziyue/facial-paralysis-h200/releases/"
        "neuroface-action-capacity-input-v1"
    )
    output_source = (
        "/home/ssh-ziyue/facial-paralysis-h200/releases/"
        "neuroface-action-capacity-output-v1"
    )
    expected_mounts = [
        {
            "type": "bind", "source": input_source,
            "destination": "/neuroface-input", "mode": "",
            "rw": False, "propagation": "rprivate",
        },
        {
            "type": "bind", "source": output_source,
            "destination": "/neuroface-output", "mode": "",
            "rw": True, "propagation": "rprivate",
        },
    ]
    expected_keys = {
        "schema_version", "host_instance_id", "gpu_model", "container_user",
        "runtime_tmpfs", "container_image_id",
        "image_id_commitment_sha256", "docker_inspect_mounts_sha256",
        "input_release", "output_release", "mounts", "nested_mounts",
        "protected_mounts",
    }
    input_value = value.get("input_release")
    output_value = value.get("output_release")
    input_identity = (
        input_value.get("identity") if isinstance(input_value, dict) else None
    )
    output_identity = (
        output_value.get("identity") if isinstance(output_value, dict) else None
    )

    def valid_identity(identity: object) -> bool:
        return bool(
            isinstance(identity, dict)
            and set(identity) == {"device", "inode", "mode", "uid", "gid"}
            and all(
                type(identity[key]) is int
                for key in ("device", "inode", "mode", "uid", "gid")
            )
            and identity["device"] >= 0
            and identity["inode"] > 0
            and stat.S_ISDIR(identity["mode"])
            and stat.S_IMODE(identity["mode"]) == 0o700
            and identity["uid"] >= 0
            and identity["gid"] >= 0
        )

    expected_input = {
        "id": "neuroface-action-capacity-input-v1",
        "source": input_source,
        "tree_sha256": input_value.get("tree_sha256")
        if isinstance(input_value, dict) else None,
        "identity": input_identity,
    }
    expected_output = {
        "id": "neuroface-action-capacity-output-v1",
        "source": output_source,
        "prestart_tree_sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "identity": output_identity,
    }
    canonical_mounts = _canonical_signed_payload(expected_mounts)
    if (
        set(value) != expected_keys
        or value.get("schema_version")
        != "neuroface_action_capacity_host_audit_payload_v1"
        or value.get("host_instance_id") != "computeinstance-e00saxxvybxg7qvj0s"
        or value.get("gpu_model") != "NVIDIA H200"
        or value.get("container_user") != "1001:1001"
        or value.get("runtime_tmpfs") != {
            "/tmp": "rw,nosuid,nodev,noexec,size=64m,mode=1777",
        }
        or not isinstance(value.get("container_image_id"), str)
        or _IMAGE_ID.fullmatch(str(value["container_image_id"])) is None
        or value.get("image_id_commitment_sha256") != hashlib.sha256(
            (str(value.get("container_image_id")) + "\n").encode("ascii")
        ).hexdigest()
        or value.get("mounts") != expected_mounts
        or value.get("docker_inspect_mounts_sha256")
        != hashlib.sha256(canonical_mounts).hexdigest()
        or value.get("input_release") != expected_input
        or not isinstance(expected_input["tree_sha256"], str)
        or _SHA256.fullmatch(str(expected_input["tree_sha256"])) is None
        or not valid_identity(input_identity)
        or value.get("output_release") != expected_output
        or not valid_identity(output_identity)
        or value.get("nested_mounts") != 0
        or value.get("protected_mounts") != 0
    ):
        raise ValueError("signed host audit payload differs from the frozen mount contract")
    return _VerifiedHostAttestation(sha256=observed, payload=dict(value))


def _validate_runtime_identity(attestation: _VerifiedHostAttestation) -> None:
    if not isinstance(attestation, _VerifiedHostAttestation):
        raise ValueError("runtime identity requires verified host evidence")
    if (
        attestation.payload.get("container_user") != "1001:1001"
        or os.getuid() != 1001
        or os.getgid() != 1001
    ):
        raise ValueError("live container UID:GID differs from signed 1001:1001")


def _validate_live_release_boundaries(
    attestation: _VerifiedHostAttestation,
    *,
    input_root: Path,
    output_root: Path,
) -> None:
    """Bind the signed host evidence to the actual mounted release inodes."""
    if not isinstance(attestation, _VerifiedHostAttestation):
        raise ValueError("live release validation requires verified host evidence")
    signed_input = attestation.payload.get("input_release")
    signed_output = attestation.payload.get("output_release")
    if not isinstance(signed_input, dict) or not isinstance(signed_output, dict):
        raise ValueError("verified host release evidence is malformed")
    input_identity = signed_input.get("identity")
    output_identity = signed_output.get("identity")
    if (
        _directory_identity(input_root) != input_identity
        or _directory_identity(output_root) != output_identity
    ):
        raise ValueError("live mounted release inode differs from signed host evidence")
    observed_input_tree = _validate_frozen_input_release(
        input_root, require_attestation_empty=False,
    )
    observed_output_tree = _release_tree_commitment(output_root)
    if (
        observed_input_tree != signed_input.get("tree_sha256")
        or observed_output_tree != signed_output.get("prestart_tree_sha256")
    ):
        raise ValueError("live mounted release tree differs from signed host evidence")
    if (
        _directory_identity(input_root) != input_identity
        or _directory_identity(output_root) != output_identity
    ):
        raise ValueError("live mounted release inode changed during validation")


def _validate_container_mountinfo(
    payload: bytes,
    *,
    attestation: _VerifiedHostAttestation,
) -> None:
    """Bind signed host sources to live mount roots and reject nested mounts."""
    if not isinstance(attestation, _VerifiedHostAttestation):
        raise ValueError("live mount validation requires a verified host attestation")
    expected_mounts = {
        str(mount["destination"]): mount
        for mount in attestation.payload["mounts"]
    }
    if set(expected_mounts) != {
        os.fspath(_NEUROFACE_INPUT_ROOT), os.fspath(_NEUROFACE_OUTPUT_ROOT),
    }:
        raise ValueError("verified host attestation mount destinations drifted")
    if type(payload) is not bytes or not payload or b"palsynet" in payload.lower():
        raise ValueError("process mount table contains protected or malformed evidence")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("process mount table is not bounded ASCII evidence") from exc
    observed: dict[str, list[tuple[str, set[str]]]] = {
        boundary: [] for boundary in expected_mounts
    }
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            raise ValueError("process mount table contains a malformed record")
        root = fields[3]
        mountpoint = fields[4]
        options = set(fields[5].split(","))
        for boundary in tuple(observed):
            if mountpoint == boundary:
                observed[boundary].append((root, options))
            elif mountpoint.startswith(boundary + "/"):
                raise ValueError("a nested mount exists inside a frozen data boundary")
    for boundary, records in observed.items():
        if len(records) != 1:
            raise ValueError("a frozen data mount is missing or duplicated")
        root, options = records[0]
        expected = expected_mounts[boundary]
        if root != expected["source"]:
            raise ValueError("live mount root differs from the signed host source")
        required_option = "rw" if expected["rw"] else "ro"
        if required_option not in options:
            raise ValueError("live mount access differs from the signed host contract")


def _require_lexical_descendant(path: Path, root: Path, *, allow_equal: bool) -> None:
    candidate = Path(path)
    if not candidate.is_absolute() or not root.is_absolute():
        raise ValueError("container boundary paths must be absolute")
    normalized = Path(os.path.abspath(candidate))
    if normalized != candidate:
        raise ValueError("container boundary paths must be lexically canonical")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes its fixed container boundary") from exc
    if not allow_equal and not relative.parts:
        raise ValueError("path must name an artifact below its container boundary")


def _validate_formal_output_root(path: Path) -> None:
    _require_lexical_descendant(path, _NEUROFACE_OUTPUT_ROOT, allow_equal=False)
    if Path(path).parent != _NEUROFACE_OUTPUT_ROOT:
        raise ValueError("formal output must be one direct release below its fixed mount")


def _primary_manifest_rows(private_manifest_bytes: bytes) -> list[dict[str, object]]:
    manifest = _json_bytes(private_manifest_bytes)
    records = manifest.get("records")
    if (
        manifest.get("schema_version") != "neuroface_external_private_manifest_v1"
        or manifest.get("dataset") != "Toronto_NeuroFace_v1"
        or manifest.get("primary_tasks") != list(PRIMARY_TASKS)
        or not isinstance(records, list)
        or len(records) != 261
    ):
        raise ValueError("private manifest differs from the frozen NeuroFace inventory")
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("private manifest records must be objects")
        recording_id = raw.get("recording_id")
        if not isinstance(recording_id, str) or recording_id in seen:
            raise ValueError("private recording identities are malformed or duplicated")
        seen.add(recording_id)
        if raw.get("task") in PRIMARY_TASKS:
            if raw.get("cohort") not in {"als", "healthy_control", "post_stroke"}:
                raise ValueError("primary recording cohort differs from the freeze")
            selected.append(raw)
    if len(selected) != 108:
        raise ValueError("the frozen private manifest must contain 108 primary recordings")
    return sorted(selected, key=lambda row: str(row["recording_id"]))


def _build_dataset_from_authoritative_bytes(
    private_manifest_bytes: bytes,
    collection_manifest_bytes: bytes,
    cache_payload_for_recording: Callable[[str], bytes],
) -> tuple[ActionCapacityDataset, str]:
    """Derive 18D rows only through the authority-bound Task 1/2 APIs."""
    if hashlib.sha256(collection_manifest_bytes).hexdigest() != (
        PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256
    ):
        raise ValueError("collection manifest differs from its activated pin")
    original: list[np.ndarray] = []
    mirrored: list[np.ndarray] = []
    participant_ids: list[str] = []
    tasks: list[str] = []
    cohorts: list[str] = []
    cache_commitments: list[tuple[str, str]] = []
    for row in _primary_manifest_rows(private_manifest_bytes):
        recording_id = str(row["recording_id"])
        video_sha256 = str(row.get("video_sha256"))
        binding = validate_neuroface_task_binding(
            private_manifest_bytes,
            recording_id=recording_id,
            decoded_recording_sha256=video_sha256,
        )
        cache_payload = cache_payload_for_recording(recording_id)
        if type(cache_payload) is not bytes:
            raise ValueError("cache reader must return the exact immutable bytes")
        capacity = neuroface_action_capacity_feature_vector(
            binding,
            cache_payload,
            collection_manifest_bytes,
            decoded_recording_sha256=video_sha256,
        )
        original.append(capacity)
        mirrored.append(mirror_action_capacity_features(capacity))
        participant_ids.append(str(row.get("participant_id")))
        tasks.append(binding.task_label)
        cohorts.append(str(row.get("cohort")))
        cache_commitments.append((recording_id, hashlib.sha256(cache_payload).hexdigest()))
    encoded_commitments = (
        json.dumps(cache_commitments, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    return ActionCapacityDataset(
        original_features=np.stack(original).astype(np.float64, copy=False),
        mirrored_features=np.stack(mirrored).astype(np.float64, copy=False),
        participant_ids=np.asarray(participant_ids, dtype=object),
        tasks=np.asarray(tasks, dtype=object),
        cohorts=np.asarray(cohorts, dtype=object),
    ), hashlib.sha256(encoded_commitments).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing a release artifact")
        view = view[written:]


def _write_json_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    encoded = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".capacity-json.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        os.chmod(output, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _private_oof_bytes(
    *,
    participant_ids: np.ndarray,
    cohorts: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    task_scores: np.ndarray,
    participant_scores: np.ndarray,
    original_probabilities: np.ndarray,
    mirrored_probabilities: np.ndarray,
) -> bytes:
    stream = io.BytesIO()
    np.savez(
        stream,
        participant_ids=np.asarray(participant_ids, dtype="U68"),
        cohorts=np.asarray(cohorts, dtype="U15"),
        labels=np.asarray(labels, dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
        task_names=np.asarray(PRIMARY_TASKS, dtype="U10"),
        task_scores=np.asarray(task_scores, dtype=np.float64),
        participant_scores=np.asarray(participant_scores, dtype=np.float64),
        original_probabilities=np.asarray(original_probabilities, dtype=np.float64),
        mirrored_probabilities=np.asarray(mirrored_probabilities, dtype=np.float64),
    )
    return stream.getvalue()


def _write_private_oof_no_overwrite(
    path: Path,
    *,
    participant_ids: np.ndarray,
    cohorts: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    task_scores: np.ndarray,
    participant_scores: np.ndarray,
    original_probabilities: np.ndarray,
    mirrored_probabilities: np.ndarray,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    encoded = _private_oof_bytes(
        participant_ids=participant_ids,
        cohorts=cohorts,
        labels=labels,
        folds=folds,
        task_scores=task_scores,
        participant_scores=participant_scores,
        original_probabilities=original_probabilities,
        mirrored_probabilities=mirrored_probabilities,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".capacity-oof.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        os.chmod(output, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_leaf_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise ValueError("release transaction members must be single path components")


def _open_secure_directory_fd(path: Path) -> int:
    """Open and identity-check a directory before any release transaction."""
    _reject_palsynet_path(path)
    _reject_symlink_components(path)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError("release parent must be an existing non-symlink directory") from exc
    try:
        opened = os.fstat(descriptor)
        leaf = os.lstat(path)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_mode)
            != (leaf.st_dev, leaf.st_ino, leaf.st_mode)
        ):
            raise ValueError("release parent identity changed while it was opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_bytes_at_no_overwrite(
    parent_fd: int,
    name: str,
    payload: bytes,
) -> str:
    _validate_leaf_name(name)
    if type(payload) is not bytes or not payload:
        raise ValueError("release artifact bytes must be non-empty")
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError:
        raise
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _write_json_at_no_overwrite(
    parent_fd: int,
    name: str,
    payload: Mapping[str, object],
) -> str:
    return _write_bytes_at_no_overwrite(parent_fd, name, _canonical_json_bytes(payload))


def _rename_directory_no_replace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a sibling directory while holding its parent inode."""
    _validate_leaf_name(source_name)
    _validate_leaf_name(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            parent_fd, source_bytes, parent_fd, destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif hasattr(libc, "renameatx_np"):
        function = libc.renameatx_np
        function.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            parent_fd, source_bytes, parent_fd, destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise RuntimeError("platform lacks an atomic no-replace directory rename")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"refusing to overwrite {destination_name}")
        raise OSError(error, os.strerror(error), destination_name)


def _remove_staging_release(parent_fd: int, staging_name: str) -> None:
    """Remove only the fixed members of an unpublished staging directory."""
    try:
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return
    try:
        try:
            private_fd = os.open(
                "private",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=staging_fd,
            )
        except FileNotFoundError:
            private_fd = None
        if private_fd is not None:
            try:
                try:
                    os.unlink("oof_scores.npz", dir_fd=private_fd)
                except FileNotFoundError:
                    pass
            finally:
                os.close(private_fd)
            os.rmdir("private", dir_fd=staging_fd)
        for name in ("report.json", "FINALIZATION.json"):
            try:
                os.unlink(name, dir_fd=staging_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(staging_fd)
    os.rmdir(staging_name, dir_fd=parent_fd)


def _write_release_atomically(
    output_root: Path,
    report: Mapping[str, object],
    result,
) -> str:
    """Stage every artifact, finalize it, then publish the whole release once."""
    output = Path(output_root)
    if not output.is_absolute():
        raise ValueError("release output must be absolute")
    parent = output.parent
    destination_name = output.name
    _validate_leaf_name(destination_name)
    parent_fd = _open_secure_directory_fd(parent)
    staging_name: str | None = None
    staging_fd: int | None = None
    published = False
    try:
        try:
            os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("capacity release output already exists")
        for _ in range(16):
            candidate = f".{destination_name}.staging-{secrets.token_hex(8)}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            staging_name = candidate
            break
        if staging_name is None:
            raise FileExistsError("could not allocate a unique release staging directory")
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        os.mkdir("private", mode=0o700, dir_fd=staging_fd)
        private_bytes = _private_oof_bytes(
            participant_ids=result.participant_ids,
            cohorts=result.cohorts,
            labels=result.labels,
            folds=result.fold_assignments,
            task_scores=result.task_scores,
            participant_scores=result.participant_scores,
            original_probabilities=result.original_probabilities,
            mirrored_probabilities=result.mirrored_probabilities,
        )
        private_fd = os.open(
            "private",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=staging_fd,
        )
        try:
            private_sha = _write_bytes_at_no_overwrite(
                private_fd, "oof_scores.npz", private_bytes
            )
            os.fsync(private_fd)
        finally:
            os.close(private_fd)
        report_sha = _write_json_at_no_overwrite(
            staging_fd, "report.json", report
        )
        finalization = {
            "schema_version": "neuroface_action_capacity_finalization_v1",
            "complete": True,
            "files": {
                "private_oof_sha256": private_sha,
                "public_report_sha256": report_sha,
            },
        }
        _write_json_at_no_overwrite(
            staging_fd, "FINALIZATION.json", finalization
        )
        os.fsync(staging_fd)
        _rename_directory_no_replace_at(
            parent_fd, staging_name, destination_name
        )
        published = True
        os.fsync(parent_fd)
        return report_sha
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        try:
            if not published and staging_name is not None:
                _remove_staging_release(parent_fd, staging_name)
        finally:
            os.close(parent_fd)


def _implementation_components_sha256() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _IMPLEMENTATION_FILES
    }


def _implementation_digest(components: Mapping[str, str]) -> str:
    encoded = (
        json.dumps(components, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256() -> str:
    return _implementation_digest(_implementation_components_sha256())


def _verify_implementation_unchanged(initial: Mapping[str, str]) -> str:
    before = dict(initial)
    after = _implementation_components_sha256()
    if before != after:
        raise ValueError("implementation source changed during the formal run")
    return _implementation_digest(before)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _verify_h200_runtime()
    implementation_before = _implementation_components_sha256()
    input_audit = _InputAccessAudit()
    for path in (
        args.private_manifest, args.collection_manifest, args.cache_root,
        args.dependency_lock, args.mount_attestation, args.output_root,
    ):
        _reject_palsynet_path(path, audit=input_audit)
    for path in (
        args.private_manifest, args.collection_manifest, args.cache_root,
        args.mount_attestation,
    ):
        _require_lexical_descendant(path, _NEUROFACE_INPUT_ROOT, allow_equal=False)
    _require_lexical_descendant(
        args.dependency_lock, PROJECT_ROOT, allow_equal=False
    )
    _validate_formal_output_root(args.output_root)
    mount_bytes, mount_sha = _read_exact_regular_file(
        args.mount_attestation, maximum_bytes=_MAX_MOUNT_ATTESTATION_BYTES,
        audit=input_audit,
    )
    host_attestation = _validate_mount_attestation(mount_bytes)
    if host_attestation.sha256 != mount_sha:
        raise ValueError("verified host attestation commitment changed")
    _validate_runtime_identity(host_attestation)
    input_audit.audit_process_mount_table(host_attestation)
    _validate_live_release_boundaries(
        host_attestation,
        input_root=_NEUROFACE_INPUT_ROOT,
        output_root=_NEUROFACE_OUTPUT_ROOT,
    )
    _reject_symlink_components(args.cache_root)
    if not args.cache_root.is_dir():
        raise ValueError("cache root must be a non-symlink directory")
    private_bytes, private_sha = _read_exact_regular_file(
        args.private_manifest, maximum_bytes=_MAX_PRIVATE_MANIFEST_BYTES,
        audit=input_audit,
    )
    collection_bytes, collection_sha = _read_exact_regular_file(
        args.collection_manifest, maximum_bytes=_MAX_COLLECTION_MANIFEST_BYTES,
        audit=input_audit,
    )
    _, dependency_sha = _read_exact_regular_file(
        args.dependency_lock, maximum_bytes=_MAX_DEPENDENCY_LOCK_BYTES,
        audit=input_audit,
    )
    if dependency_sha != PINNED_DEPENDENCY_LOCK_SHA256:
        raise ValueError("dependency lock differs from the frozen H200 environment")

    def cache_reader(recording_id: str) -> bytes:
        payload, _ = _read_exact_regular_file(
            args.cache_root / f"{recording_id}.npz", maximum_bytes=_MAX_CACHE_BYTES,
            audit=input_audit,
        )
        return payload

    started = time.monotonic()
    dataset, primary_cache_collection_sha = _build_dataset_from_authoritative_bytes(
        private_bytes, collection_bytes, cache_reader
    )
    result = evaluate_action_capacity_oof(dataset)
    implementation_sha = _verify_implementation_unchanged(implementation_before)
    action_audit = input_audit.formal_action_audit()
    elapsed = time.monotonic() - started
    provenance = {
        "private_manifest_sha256": private_sha,
        "collection_manifest_sha256": collection_sha,
        "primary_cache_collection_sha256": primary_cache_collection_sha,
        "implementation_sha256": implementation_sha,
        "dependency_lock_sha256": dependency_sha,
        "mount_attestation_sha256": mount_sha,
    }
    runtime = {
        "host_class": "nebius_h200", "device_class": "cpu", "seconds": elapsed,
    }
    report = build_public_report(
        result, provenance=provenance, audit=action_audit, runtime=runtime,
    )
    validate_public_report(
        report,
        result=result,
        expected_provenance=provenance,
        expected_runtime=runtime,
        expected_audit=action_audit,
    )
    report_sha = _write_release_atomically(args.output_root, report, result)
    print(json.dumps({
        "schema_version": "neuroface_action_capacity_receipt_v1",
        "report_sha256": report_sha,
        "bootstrap_repeats": 5000,
        "protected_data_accesses": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
