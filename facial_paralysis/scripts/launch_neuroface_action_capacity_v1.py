#!/usr/bin/env python3
"""Create, attest, and start the one frozen NeuroFace capacity container.

Post-build activation is deliberately outside the image: a trusted host operator
must write the exact ``docker image inspect --format '{{.Id}}'`` result plus one
LF to ``HOST_IMAGE_ID_COMMITMENT_PATH`` with mode 0600.  The launcher safe-reads
that immutable-image commitment and refuses a tag whose inspected ID differs.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping


HOST_INPUT_RELEASE_ROOT = Path(
    "/home/ssh-ziyue/facial-paralysis-h200/releases/"
    "neuroface-action-capacity-input-v1"
)
HOST_OUTPUT_RELEASE_ROOT = Path(
    "/home/ssh-ziyue/facial-paralysis-h200/releases/"
    "neuroface-action-capacity-output-v1"
)
HOST_PRIVATE_KEY_PATH = Path(
    "/home/ssh-ziyue/.config/facial-paralysis/"
    "action-capacity-attestation-v1/private-ed25519.pem"
)
# Populated once after the v1.4 image build. This host-only activation is never
# copied into the image, so pinning the image ID cannot become self-referential.
HOST_IMAGE_ID_COMMITMENT_PATH = Path(
    "/home/ssh-ziyue/facial-paralysis-h200/activations/"
    "neuroface-action-capacity-v1.4-image-id.txt"
)
CONTAINER_IMAGE = "facial-paralysis-neuroface:v1.4"
CONTAINER_NAME = "neuroface-action-capacity-v1"
CONTAINER_USER = "1001:1001"
HOST_INSTANCE_ID = "computeinstance-e00saxxvybxg7qvj0s"
GPU_MODEL = "NVIDIA H200"
PRIVATE_MANIFEST_SHA256 = (
    "235d2af2f3f4507b4ec858ff8dd9ff949d7f19e0d3656cbf5dcc0218648da07b"
)
COLLECTION_MANIFEST_SHA256 = (
    "07527c33fe0e35d34a554f7baccd49e9e692c4588b76aa6392ccad71c122bb17"
)
ATTESTATION_RELATIVE_PATH = Path("attestation/host_attestation.json")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def _unique_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("host audit JSON contains a duplicate key")
        value[key] = child
    return value


def _strict_json(payload: bytes) -> object:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("host audit input is not strict UTF-8 JSON") from exc


def _canonical_payload(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _canonical_envelope(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
    ):
        raise ValueError("host audit file request is malformed")
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("host audit input is not a regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError("host audit input file identity is unsafe")
        chunks = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        leaf = os.lstat(candidate)
        identity = lambda metadata: (
            metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink,
            metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
        )
        if (
            len(payload) != before.st_size
            or len(payload) > maximum_bytes
            or identity(before) != identity(after)
            or identity(after) != identity(leaf)
        ):
            raise ValueError("host audit input changed while it was read")
        return payload
    finally:
        os.close(descriptor)


def _release_tree_commitment(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset({ATTESTATION_RELATIVE_PATH.as_posix()}),
) -> str:
    """Commit every owner-private member while rejecting aliases and symlinks."""
    release = Path(root)
    metadata = os.lstat(release)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("release root must be an owner-private directory")
    entries: list[dict[str, object]] = []
    stack = [release]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as stream:
            children = sorted(stream, key=lambda child: child.name, reverse=True)
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(release).as_posix()
            if relative in excluded:
                continue
            if "palsynet" in relative.casefold():
                raise ValueError("protected-looking member exists in the NeuroFace release")
            child_metadata = os.lstat(path)
            if stat.S_ISLNK(child_metadata.st_mode):
                raise ValueError("release trees cannot contain symlinks")
            if stat.S_ISDIR(child_metadata.st_mode):
                if stat.S_IMODE(child_metadata.st_mode) != 0o700:
                    raise ValueError("release directories must be owner-private")
                entries.append({"mode": "0700", "path": relative, "type": "directory"})
                stack.append(path)
            elif stat.S_ISREG(child_metadata.st_mode):
                if (
                    stat.S_IMODE(child_metadata.st_mode) != 0o600
                    or child_metadata.st_nlink != 1
                ):
                    raise ValueError("release files must be owner-private single links")
                payload = _read_regular_file(path, maximum_bytes=64 * 1024 * 1024)
                entries.append({
                    "mode": "0600", "path": relative, "type": "file",
                    "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                })
            else:
                raise ValueError("release trees may contain only directories and files")
    entries.sort(key=lambda entry: str(entry["path"]))
    return hashlib.sha256(_canonical_payload(entries)).hexdigest()


def _directory_identity(path: Path) -> dict[str, int]:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("signed release identity requires an owner-private directory")
    return {
        "device": int(metadata.st_dev), "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode), "uid": int(metadata.st_uid),
        "gid": int(metadata.st_gid),
    }


def _read_image_id_commitment(path: Path) -> str:
    commitment_path = Path(path)
    metadata = os.lstat(commitment_path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("post-build image-ID commitment identity is unsafe")
    payload = _read_regular_file(commitment_path, maximum_bytes=128)
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("post-build image-ID commitment is not ASCII") from exc
    if not value.endswith("\n") or _IMAGE_ID.fullmatch(value[:-1]) is None:
        raise ValueError("post-build image-ID commitment is not exact sha256:<64hex> LF")
    return value[:-1]


def _validate_frozen_input_release(
    root: Path,
    *,
    require_attestation_empty: bool = True,
) -> str:
    release = Path(root)
    _directory_identity(release)
    children = {child.name for child in release.iterdir()}
    if children != {"attestation", "cache", "private"}:
        raise ValueError("frozen input release has an unexpected top-level member")
    attestation_members = {path.name for path in (release / "attestation").iterdir()}
    if require_attestation_empty:
        if attestation_members:
            raise FileExistsError("host attestation already exists; refusing a second launch")
    elif attestation_members != {ATTESTATION_RELATIVE_PATH.name}:
        raise ValueError("live input attestation directory differs from the signed release")
    private_path = release / "private" / "participant_manifest.json"
    collection_path = release / "cache" / "collection_manifest.json"
    private_bytes = _read_regular_file(private_path, maximum_bytes=8 * 1024 * 1024)
    collection_bytes = _read_regular_file(collection_path, maximum_bytes=8 * 1024 * 1024)
    if hashlib.sha256(private_bytes).hexdigest() != PRIVATE_MANIFEST_SHA256:
        raise ValueError("private manifest differs from its frozen NeuroFace pin")
    if hashlib.sha256(collection_bytes).hexdigest() != COLLECTION_MANIFEST_SHA256:
        raise ValueError("collection manifest differs from its frozen NeuroFace pin")
    collection = _strict_json(collection_bytes)
    if not isinstance(collection, dict) or not isinstance(collection.get("records"), list):
        raise ValueError("collection manifest records are malformed")
    retained: dict[str, str] = {}
    excluded = 0
    for record in collection["records"]:
        if not isinstance(record, dict):
            raise ValueError("collection manifest record is malformed")
        recording_id = record.get("recording_id")
        status_value = record.get("status")
        if not isinstance(recording_id, str) or _RECORDING_ID.fullmatch(recording_id) is None:
            raise ValueError("collection recording identity is malformed")
        if status_value == "retained":
            cache_sha = record.get("cache_sha256")
            if (
                not isinstance(cache_sha, str) or _SHA256.fullmatch(cache_sha) is None
                or recording_id in retained
            ):
                raise ValueError("retained cache commitment is malformed or duplicated")
            retained[recording_id] = cache_sha
        elif status_value == "excluded":
            excluded += 1
        else:
            raise ValueError("collection status differs from retained/excluded")
    if len(collection["records"]) != 261 or len(retained) != 231 or excluded != 30:
        raise ValueError("collection counts differ from the frozen 261/231/30 release")
    cache_files = {
        path.name for path in (release / "cache").iterdir()
        if path.name != "collection_manifest.json"
    }
    expected_files = {f"{recording_id}.npz" for recording_id in retained}
    if cache_files != expected_files:
        raise ValueError("cache directory differs from the 231 retained records")
    for recording_id, expected_sha in retained.items():
        cache_payload = _read_regular_file(
            release / "cache" / f"{recording_id}.npz",
            maximum_bytes=64 * 1024 * 1024,
        )
        if hashlib.sha256(cache_payload).hexdigest() != expected_sha:
            raise ValueError("retained cache differs from the collection commitment")
    if {path.name for path in (release / "private").iterdir()} != {
        "participant_manifest.json",
    }:
        raise ValueError("private input directory contains an unexpected member")
    return _release_tree_commitment(release)


def _validated_mount_projection(
    inspect: Mapping[str, object],
) -> tuple[list[dict[str, object]], str]:
    raw_mounts = inspect.get("Mounts")
    if not isinstance(raw_mounts, list) or len(raw_mounts) != 2:
        raise ValueError("container must have exactly two Docker mounts")
    projected = []
    for raw in raw_mounts:
        if not isinstance(raw, dict):
            raise ValueError("Docker mount record is malformed")
        projected.append({
            "type": raw.get("Type"), "source": raw.get("Source"),
            "destination": raw.get("Destination"), "mode": raw.get("Mode"),
            "rw": raw.get("RW"), "propagation": raw.get("Propagation"),
        })
    projected.sort(key=lambda mount: str(mount["destination"]))
    expected = [
        {
            "type": "bind", "source": os.fspath(HOST_INPUT_RELEASE_ROOT),
            "destination": "/neuroface-input", "mode": "", "rw": False,
            "propagation": "rprivate",
        },
        {
            "type": "bind", "source": os.fspath(HOST_OUTPUT_RELEASE_ROOT),
            "destination": "/neuroface-output", "mode": "", "rw": True,
            "propagation": "rprivate",
        },
    ]
    if projected != expected or "palsynet" in json.dumps(projected).casefold():
        raise ValueError("Docker mounts differ from the exact NeuroFace-only contract")
    canonical = _canonical_payload(projected)
    return projected, hashlib.sha256(canonical).hexdigest()


def _docker(args: list[str], *, timeout: float):
    return subprocess.run(
        ["sudo", "-n", "docker", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )


def _verify_h200_host() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5.0, shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("host GPU evidence is unavailable") from exc
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if not names.intersection({"NVIDIA H200", "NVIDIA H200 NVL"}):
        raise ValueError("formal host is not an NVIDIA H200")


def _docker_create_command() -> list[str]:
    return [
        "container", "create", "--name", CONTAINER_NAME,
        "--gpus", "all", "--network", "none", "--read-only",
        "--user", CONTAINER_USER,
        "--mount", (
            f"type=bind,src={HOST_INPUT_RELEASE_ROOT},"
            "dst=/neuroface-input,readonly"
        ),
        "--mount", (
            f"type=bind,src={HOST_OUTPUT_RELEASE_ROOT},"
            "dst=/neuroface-output"
        ),
        "--workdir", "/workspace/facial_paralysis",
        CONTAINER_IMAGE,
        "python", "scripts/run_neuroface_action_capacity_v1.py",
        "--private-manifest", "/neuroface-input/private/participant_manifest.json",
        "--collection-manifest", "/neuroface-input/cache/collection_manifest.json",
        "--cache-root", "/neuroface-input/cache",
        "--dependency-lock", "/workspace/facial_paralysis/environment/neuroface_h200_v1.lock",
        "--mount-attestation", "/neuroface-input/attestation/host_attestation.json",
        "--output-root", "/neuroface-output/action-capacity-v1",
    ]


def _sign_payload(payload: bytes, *, private_key_path: Path) -> bytes:
    key = Path(private_key_path)
    metadata = os.lstat(key)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("host Ed25519 private key identity or permissions are unsafe")
    with tempfile.TemporaryDirectory(prefix="neuroface-host-audit-sign-") as temporary:
        root = Path(temporary)
        os.chmod(root, 0o700)
        message = root / "payload.json"
        signature = root / "signature.bin"
        descriptor = os.open(
            message, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while staging the signed host payload")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            subprocess.run(
                [
                    "openssl", "pkeyutl", "-sign", "-inkey", os.fspath(key),
                    "-rawin", "-in", os.fspath(message), "-out", os.fspath(signature),
                ],
                check=True, capture_output=True, timeout=5.0, shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("host Ed25519 signing failed") from exc
        signed = _read_regular_file(signature, maximum_bytes=1024)
        if len(signed) != 64:
            raise ValueError("host Ed25519 signer returned a malformed signature")
        return signed


def _write_attestation_no_overwrite(path: Path, payload: bytes) -> None:
    output = Path(path)
    directory_fd = os.open(
        output.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _write_attestation_at_no_overwrite(directory_fd, output.name, payload)
    finally:
        os.close(directory_fd)


def _write_attestation_at_no_overwrite(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    if name != ATTESTATION_RELATIVE_PATH.name or type(payload) is not bytes or not payload:
        raise ValueError("host attestation publication request is malformed")
    temporary = f".host-attestation-{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    linked = False
    try:
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while publishing host attestation")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temporary, name,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(directory_fd)
    except BaseException:
        if linked:
            os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        raise
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _validated_receipt(receipt: str) -> dict[str, object]:
    try:
        value = json.loads(receipt, object_pairs_hook=_unique_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("container returned a malformed path-free receipt") from exc
    expected_keys = {
        "schema_version", "report_sha256", "bootstrap_repeats",
        "protected_data_accesses",
    }
    if (
        not isinstance(value, dict) or set(value) != expected_keys
        or value.get("schema_version") != "neuroface_action_capacity_receipt_v1"
        or not isinstance(value.get("report_sha256"), str)
        or _SHA256.fullmatch(str(value["report_sha256"])) is None
        or value.get("bootstrap_repeats") != 5000
        or value.get("protected_data_accesses") != 0
    ):
        raise ValueError("container receipt differs from the closed formal schema")
    return value


def _validate_formal_output_release(
    output_root: Path,
    *,
    signed_identity: Mapping[str, object],
    receipt: Mapping[str, object],
) -> None:
    root = Path(output_root)
    if _directory_identity(root) != dict(signed_identity):
        raise ValueError("host output release identity changed across container execution")
    if {path.name for path in root.iterdir()} != {"action-capacity-v1"}:
        raise ValueError("formal output root differs from its one-release contract")
    release = root / "action-capacity-v1"
    if {path.name for path in release.iterdir()} != {
        "FINALIZATION.json", "private", "report.json",
    } or {path.name for path in (release / "private").iterdir()} != {
        "oof_scores.npz",
    }:
        raise ValueError("formal output release contains missing or extra artifacts")
    report_bytes = _read_regular_file(release / "report.json", maximum_bytes=8 * 1024 * 1024)
    private_bytes = _read_regular_file(
        release / "private" / "oof_scores.npz", maximum_bytes=8 * 1024 * 1024,
    )
    finalization_bytes = _read_regular_file(
        release / "FINALIZATION.json", maximum_bytes=1024 * 1024,
    )
    finalization = _strict_json(finalization_bytes)
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    private_sha = hashlib.sha256(private_bytes).hexdigest()
    expected_finalization = {
        "schema_version": "neuroface_action_capacity_finalization_v1",
        "complete": True,
        "files": {
            "private_oof_sha256": private_sha,
            "public_report_sha256": report_sha,
        },
    }
    if finalization != expected_finalization or receipt.get("report_sha256") != report_sha:
        raise ValueError("formal output hashes differ from finalization or receipt")


def _launch_once() -> str:
    _verify_h200_host()
    pinned_image_id = _read_image_id_commitment(HOST_IMAGE_ID_COMMITMENT_PATH)
    input_tree = _validate_frozen_input_release(HOST_INPUT_RELEASE_ROOT)
    output_tree = _release_tree_commitment(HOST_OUTPUT_RELEASE_ROOT)
    empty_tree = hashlib.sha256(b"[]\n").hexdigest()
    if output_tree != empty_tree:
        raise FileExistsError("formal output release must be empty before Docker create")
    input_identity = _directory_identity(HOST_INPUT_RELEASE_ROOT)
    output_identity = _directory_identity(HOST_OUTPUT_RELEASE_ROOT)
    attestation_directory_fd = os.open(
        HOST_INPUT_RELEASE_ROOT / "attestation",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    container_id: str | None = None
    attestation_published = False
    try:
        created = _docker(_docker_create_command(), timeout=30.0)
        container_id = created.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            raise ValueError("docker create did not return one full container identity")
        inspected = _docker(["container", "inspect", container_id], timeout=10.0)
        inspect_value = _strict_json(inspected.stdout.encode("utf-8"))
        if (
            not isinstance(inspect_value, list) or len(inspect_value) != 1
            or not isinstance(inspect_value[0], dict)
            or inspect_value[0].get("Id") != container_id
            or inspect_value[0].get("Image") != pinned_image_id
            or not isinstance(inspect_value[0].get("Config"), dict)
            or inspect_value[0]["Config"].get("User") != CONTAINER_USER
        ):
            raise ValueError(
                "docker inspect identity, image pin, or runtime user differs"
            )
        mounts, mount_sha = _validated_mount_projection(inspect_value[0])
        payload = {
        "schema_version": "neuroface_action_capacity_host_audit_payload_v1",
        "host_instance_id": HOST_INSTANCE_ID,
        "gpu_model": GPU_MODEL,
        "container_user": CONTAINER_USER,
        "container_image_id": pinned_image_id,
        "image_id_commitment_sha256": hashlib.sha256(
            (pinned_image_id + "\n").encode("ascii")
        ).hexdigest(),
        "docker_inspect_mounts_sha256": mount_sha,
        "input_release": {
            "id": "neuroface-action-capacity-input-v1",
            "source": os.fspath(HOST_INPUT_RELEASE_ROOT),
            "tree_sha256": input_tree,
            "identity": input_identity,
        },
        "output_release": {
            "id": "neuroface-action-capacity-output-v1",
            "source": os.fspath(HOST_OUTPUT_RELEASE_ROOT),
            "prestart_tree_sha256": output_tree,
            "identity": output_identity,
        },
        "mounts": mounts,
        "nested_mounts": 0,
        "protected_mounts": 0,
        }
        signed_payload = _canonical_payload(payload)
        signature = _sign_payload(signed_payload, private_key_path=HOST_PRIVATE_KEY_PATH)
        envelope = _canonical_envelope({
        "schema_version": "neuroface_action_capacity_signed_host_attestation_v1",
        "payload": payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        })
        if (
            _validate_frozen_input_release(HOST_INPUT_RELEASE_ROOT) != input_tree
            or _directory_identity(HOST_INPUT_RELEASE_ROOT) != input_identity
        ):
            raise ValueError("frozen input release changed between inspect and signing")
        if (
            _release_tree_commitment(HOST_OUTPUT_RELEASE_ROOT) != output_tree
            or _directory_identity(HOST_OUTPUT_RELEASE_ROOT) != output_identity
        ):
            raise ValueError("output release changed between inspect and signing")
        _write_attestation_at_no_overwrite(
            attestation_directory_fd, ATTESTATION_RELATIVE_PATH.name, envelope
        )
        attestation_published = True
        started = _docker(
            ["container", "start", "--attach", container_id], timeout=1800.0
        )
        receipt_text = started.stdout.strip()
        receipt = _validated_receipt(receipt_text)
        _validate_formal_output_release(
            HOST_OUTPUT_RELEASE_ROOT,
            signed_identity=output_identity,
            receipt=receipt,
        )
        return receipt_text
    except BaseException as original_error:
        cleanup_errors: list[BaseException] = []
        if container_id is not None:
            try:
                _docker(["container", "rm", "--force", container_id], timeout=30.0)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if attestation_published:
            try:
                os.unlink(ATTESTATION_RELATIVE_PATH.name, dir_fd=attestation_directory_fd)
                os.fsync(attestation_directory_fd)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError(
                "formal launch failed and cleanup could not remove all attempt state"
            ) from original_error
        raise
    finally:
        os.close(attestation_directory_fd)


def main() -> int:
    print(_launch_once())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
