"""Runner contracts for exact-byte NeuroFace capacity evaluation."""
from __future__ import annotations

import base64
import hashlib
import json
import stat
import sys
import tempfile
import os
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
import scripts.run_neuroface_action_capacity_v1 as runner  # noqa: E402
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
)


def _signed_host_payload() -> dict[str, object]:
    input_source = ("/home/ssh-ziyue/facial-paralysis-h200/releases/"
                    "neuroface-action-capacity-input-v1")
    output_source = ("/home/ssh-ziyue/facial-paralysis-h200/releases/"
                     "neuroface-action-capacity-output-v1")
    mounts = [
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
    mount_bytes = (json.dumps(
        mounts, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("ascii")
    return {
        "schema_version": "neuroface_action_capacity_host_audit_payload_v1",
        "host_instance_id": "computeinstance-e00saxxvybxg7qvj0s",
        "gpu_model": "NVIDIA H200",
        "container_user": "1001:1001",
        "container_image_id": "sha256:" + "a" * 64,
        "image_id_commitment_sha256": hashlib.sha256(
            (("sha256:" + "a" * 64) + "\n").encode("ascii")
        ).hexdigest(),
        "docker_inspect_mounts_sha256": hashlib.sha256(mount_bytes).hexdigest(),
        "input_release": {
            "id": "neuroface-action-capacity-input-v1",
            "source": input_source,
            "tree_sha256": "c" * 64,
            "identity": {
                "device": 1, "inode": 2, "mode": 0o40700, "uid": 3, "gid": 4,
            },
        },
        "output_release": {
            "id": "neuroface-action-capacity-output-v1",
            "source": output_source,
            "prestart_tree_sha256": hashlib.sha256(b"[]\n").hexdigest(),
            "identity": {
                "device": 1, "inode": 5, "mode": 0o40700, "uid": 3, "gid": 4,
            },
        },
        "mounts": mounts,
        "nested_mounts": 0,
        "protected_mounts": 0,
    }


def _signed_envelope_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps({
        "schema_version": "neuroface_action_capacity_signed_host_attestation_v1",
        "payload": payload,
        "signature_base64": base64.b64encode(b"s" * 64).decode("ascii"),
    }, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def test_cli_has_exact_inputs_and_no_tuning_surface(c: Check):
    parser = runner._parser()
    namespace = parser.parse_args([
        "--private-manifest", "/private/neuroface.json",
        "--collection-manifest", "/input/collection_manifest.json",
        "--cache-root", "/input/cache",
        "--dependency-lock", "/code/requirements.lock",
        "--mount-attestation", "/private/mount-attestation.json",
        "--output-root", "/output/capacity-v1",
    ])
    c.eq(set(vars(namespace)), {
        "private_manifest", "collection_manifest", "cache_root",
        "dependency_lock", "mount_attestation", "output_root",
    })
    for forbidden in ("--C", "--seed", "--folds", "--threshold", "--bootstrap-repeats"):
        try:
            parser.parse_args([
                "--private-manifest", "a", "--collection-manifest", "b",
                "--cache-root", "c", "--dependency-lock", "d",
                "--mount-attestation", "m", "--output-root", "e", forbidden, "1",
            ])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"runner unexpectedly accepts tuning flag {forbidden}")
    c.eq(PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
         "07527c33fe0e35d34a554f7baccd49e9e692c4588b76aa6392ccad71c122bb17")
    c.eq(runner.PINNED_DEPENDENCY_LOCK_SHA256,
         "f71f528af621e4eff83bb6a05c1fff09b0918ecda2cd9ac67979582b67767a6a")
    runner._validate_formal_output_root(Path("/neuroface-output/capacity-v1"))
    c.raises(lambda: runner._validate_formal_output_root(
        Path("/neuroface-output/nested/capacity-v1")
    ), ValueError)


def test_mount_attestation_is_exact_out_of_band_and_closed(c: Check):
    payload = _signed_host_payload()
    envelope = _signed_envelope_bytes(payload)
    original_verify = runner._verify_ed25519_signature
    runner._verify_ed25519_signature = lambda message, signature: None
    try:
        attestation = runner._validate_mount_attestation(envelope)
        c.eq(attestation.sha256, hashlib.sha256(envelope).hexdigest())
        c.eq(attestation.payload, payload)
        changed = _signed_host_payload()
        changed["protected_mounts"] = 1
        c.raises(lambda: runner._validate_mount_attestation(
            _signed_envelope_bytes(changed)
        ), ValueError)
    finally:
        runner._verify_ed25519_signature = original_verify

    source = (ROOT / "scripts" / "run_neuroface_action_capacity_v1.py").read_text()
    c.true("NEUROFACE_ACTION_CAPACITY_MOUNT_ATTESTATION_SHA256" not in source,
           "the in-container caller cannot override the host attestation pin")

    mountinfo = (
        b"20 1 0:1 / / rw,relatime - overlay overlay rw\n"
        b"21 20 8:1 /home/ssh-ziyue/facial-paralysis-h200/releases/"
        b"neuroface-action-capacity-input-v1 /neuroface-input ro,relatime - ext4 /dev/vda1 rw\n"
        b"22 20 8:1 /home/ssh-ziyue/facial-paralysis-h200/releases/"
        b"neuroface-action-capacity-output-v1 /neuroface-output rw,relatime - ext4 /dev/vda1 rw\n"
    )
    runner._validate_container_mountinfo(mountinfo, attestation=attestation)
    c.raises(lambda: runner._validate_container_mountinfo(
        mountinfo +
        b"23 21 8:1 /hidden /neuroface-input/nested ro - ext4 /dev/vda1 rw\n",
        attestation=attestation,
    ), ValueError)
    c.raises(lambda: runner._validate_container_mountinfo(
        mountinfo.replace(b"/neuroface-input ro,", b"/neuroface-input rw,"),
        attestation=attestation,
    ), ValueError)
    c.raises(lambda: runner._validate_container_mountinfo(
        mountinfo.replace(b"neuroface-action-capacity-input-v1 /neuroface-input",
                          b"untrusted-source /neuroface-input"),
        attestation=attestation,
    ), ValueError)


def test_ed25519_host_signature_and_public_key_pin_are_real(c: Check):
    public_resource = (
        ROOT / "environment" / "neuroface_action_capacity_host_audit_ed25519_public.pem"
    )
    pem = public_resource.read_bytes()
    body = b"".join(line for line in pem.splitlines() if not line.startswith(b"-----"))
    der = base64.b64decode(body, validate=True)
    c.eq(hashlib.sha256(der).hexdigest(),
         "95229c6132a163e0ad073e5f6f8b9f3bdb8c7e52a292da3097310b24c1735904")

    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        private = root / "fixture-private.pem"
        public = root / "fixture-public.pem"
        message = root / "message.bin"
        signature = root / "signature.bin"
        message.write_bytes(b"canonical host audit fixture\n")
        subprocess.run([
            "openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private),
        ], check=True, capture_output=True)
        subprocess.run([
            "openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public),
        ], check=True, capture_output=True)
        subprocess.run([
            "openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin",
            "-in", str(message), "-out", str(signature),
        ], check=True, capture_output=True)
        public_der = subprocess.run([
            "openssl", "pkey", "-pubin", "-in", str(public), "-outform", "DER",
        ], check=True, capture_output=True).stdout
        original_path = runner._HOST_ATTESTATION_PUBLIC_KEY_PATH
        original_pin = runner.PINNED_HOST_ATTESTATION_PUBLIC_KEY_DER_SHA256
        runner._HOST_ATTESTATION_PUBLIC_KEY_PATH = public
        runner.PINNED_HOST_ATTESTATION_PUBLIC_KEY_DER_SHA256 = hashlib.sha256(
            public_der
        ).hexdigest()
        original_run = runner.subprocess.run
        runner.subprocess.run = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("in-container Ed25519 verification must not depend on OpenSSL")
        )
        try:
            runner._verify_ed25519_signature(message.read_bytes(), signature.read_bytes())
            c.raises(lambda: runner._verify_ed25519_signature(
                message.read_bytes() + b"tampered", signature.read_bytes()
            ), ValueError)
        finally:
            runner.subprocess.run = original_run
            runner._HOST_ATTESTATION_PUBLIC_KEY_PATH = original_path
            runner.PINNED_HOST_ATTESTATION_PUBLIC_KEY_DER_SHA256 = original_pin


def test_h200_attestation_uses_nvidia_smi_with_a_fixed_no_shell_query(c: Check):
    calls = []

    def h200_query(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="NVIDIA H200\n", returncode=0)

    original_run = runner.subprocess.run
    try:
        runner.subprocess.run = h200_query
        runner._verify_h200_runtime()
        runner.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
            stdout="NVIDIA B200\n", returncode=0
        )
        c.raises(runner._verify_h200_runtime, ValueError)
        runner.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
            stdout="NVIDIA H200-fake\n", returncode=0
        )
        c.raises(runner._verify_h200_runtime, ValueError)
    finally:
        runner.subprocess.run = original_run
    c.eq(calls, [([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
    ], {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 5.0,
        "shell": False,
    })])


def test_runtime_uid_gid_are_bound_to_the_signed_container_user(c: Check):
    payload = _signed_host_payload()
    attestation = runner._VerifiedHostAttestation("f" * 64, payload)
    original_getuid = runner.os.getuid
    original_getgid = runner.os.getgid
    try:
        runner.os.getuid = lambda: 1001
        runner.os.getgid = lambda: 1001
        runner._validate_runtime_identity(attestation)
        runner.os.getuid = lambda: 0
        c.raises(lambda: runner._validate_runtime_identity(attestation), ValueError)
    finally:
        runner.os.getuid = original_getuid
        runner.os.getgid = original_getgid


def test_live_release_tree_and_inode_must_match_signed_host_evidence(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        input_root = root / "input"
        output_root = root / "output"
        input_root.mkdir(mode=0o700)
        output_root.mkdir(mode=0o700)
        payload = _signed_host_payload()

        def identity(path):
            metadata = os.lstat(path)
            return {
                "device": metadata.st_dev, "inode": metadata.st_ino,
                "mode": metadata.st_mode, "uid": metadata.st_uid, "gid": metadata.st_gid,
            }

        payload["input_release"]["identity"] = identity(input_root)
        payload["output_release"]["identity"] = identity(output_root)
        attestation = runner._VerifiedHostAttestation("f" * 64, payload)
        original = runner._validate_frozen_input_release
        runner._validate_frozen_input_release = lambda path, **kwargs: "c" * 64
        try:
            runner._validate_live_release_boundaries(
                attestation, input_root=input_root, output_root=output_root
            )
            saved = root / "saved-input"
            input_root.rename(saved)
            input_root.mkdir(mode=0o700)
            c.raises(lambda: runner._validate_live_release_boundaries(
                attestation, input_root=input_root, output_root=output_root
            ), ValueError)
        finally:
            runner._validate_frozen_input_release = original


def test_exact_byte_reader_rejects_symlinks_changes_and_oversize(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        source = root / "input.bin"
        source.write_bytes(b"exact-neuroface-bytes")
        payload, digest = runner._read_exact_regular_file(source, maximum_bytes=64)
        c.eq(payload, b"exact-neuroface-bytes")
        c.eq(digest, hashlib.sha256(payload).hexdigest())
        symlink = root / "link.bin"
        symlink.symlink_to(source)
        c.raises(lambda: runner._read_exact_regular_file(
            symlink, maximum_bytes=64
        ), ValueError)
        c.raises(lambda: runner._read_exact_regular_file(
            source, maximum_bytes=4
        ), ValueError)

        protected = root / "palsy-secret"
        protected.mkdir()
        target = protected / "manifest.json"
        target.write_bytes(b"do-not-read")
        innocent = root / "innocent"
        innocent.symlink_to(protected, target_is_directory=True)
        c.raises(lambda: runner._read_exact_regular_file(
            innocent / "manifest.json", maximum_bytes=64
        ), ValueError, "a parent symlink is rejected before following it")
        hardlink = root / "innocent-hardlink.json"
        os.link(target, hardlink)
        c.raises(lambda: runner._read_exact_regular_file(
            hardlink, maximum_bytes=64
        ), ValueError, "hardlink aliases are not authenticated input files")


def test_output_writers_are_owner_private_atomic_and_no_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report_path = root / "report.json"
        report = {"schema_version": "fixture", "value": 1}
        runner._write_json_no_overwrite(report_path, report)
        c.eq(stat.S_IMODE(report_path.stat().st_mode), 0o600)
        c.eq(json.loads(report_path.read_text()), report)
        c.raises(lambda: runner._write_json_no_overwrite(
            report_path, report
        ), FileExistsError)

        private_path = root / "private" / "oof.npz"
        runner._write_private_oof_no_overwrite(
            private_path,
            participant_ids=np.asarray(["grp_" + "a" * 64], dtype=object),
            cohorts=np.asarray(["als"], dtype=object),
            labels=np.asarray([1], dtype=np.int64),
            folds=np.asarray([0], dtype=np.int64),
            task_scores=np.asarray([[0.7, 0.8, 0.9]], dtype=np.float64),
            participant_scores=np.asarray([0.8], dtype=np.float64),
            original_probabilities=np.asarray([[0.6, 0.7, 0.8]], dtype=np.float64),
            mirrored_probabilities=np.asarray([[0.8, 0.9, 1.0]], dtype=np.float64),
        )
        c.eq(stat.S_IMODE(private_path.stat().st_mode), 0o600)
        c.raises(lambda: runner._write_private_oof_no_overwrite(
            private_path,
            participant_ids=np.asarray([], dtype=object), cohorts=np.asarray([], dtype=object),
            labels=np.asarray([], dtype=np.int64), folds=np.asarray([], dtype=np.int64),
            task_scores=np.empty((0, 3)), participant_scores=np.asarray([]),
            original_probabilities=np.empty((0, 3)),
            mirrored_probabilities=np.empty((0, 3)),
        ), FileExistsError)


def test_whole_release_publication_is_atomic_and_cleans_failed_staging(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        parent = Path(temporary)
        output = parent / "release-v1"
        result = SimpleNamespace(
            participant_ids=np.asarray(["grp_" + "a" * 64], dtype=object),
            cohorts=np.asarray(["als"], dtype=object),
            labels=np.asarray([1], dtype=np.int64),
            fold_assignments=np.asarray([0], dtype=np.int64),
            task_scores=np.asarray([[0.7, 0.8, 0.9]], dtype=np.float64),
            participant_scores=np.asarray([0.8], dtype=np.float64),
            original_probabilities=np.asarray([[0.6, 0.7, 0.8]], dtype=np.float64),
            mirrored_probabilities=np.asarray([[0.8, 0.9, 1.0]], dtype=np.float64),
        )
        original = runner._write_json_at_no_overwrite
        calls = {"count": 0}

        def fail_report(parent_fd, name, payload):
            calls["count"] += 1
            if name == "report.json":
                raise RuntimeError("injected report write failure")
            return original(parent_fd, name, payload)

        runner._write_json_at_no_overwrite = fail_report
        try:
            c.raises(lambda: runner._write_release_atomically(
                output, {"schema_version": "fixture"}, result
            ), RuntimeError)
        finally:
            runner._write_json_at_no_overwrite = original
        c.true(not output.exists(), "partial private output is never published")
        c.eq(list(parent.glob(".release-v1.staging-*")), [],
             "failed private staging is removed")

        runner._write_release_atomically(
            output, {"schema_version": "fixture"}, result
        )
        c.true((output / "report.json").is_file())
        c.true((output / "private" / "oof_scores.npz").is_file())
        c.true((output / "FINALIZATION.json").is_file())
        c.raises(lambda: runner._write_release_atomically(
            output, {"schema_version": "fixture"}, result
        ), FileExistsError)


def test_whole_release_holds_parent_inode_across_rename_symlink_attack(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        parent = root / "owner"
        saved_parent = root / "owner-held"
        attacker = root / "attacker"
        parent.mkdir()
        attacker.mkdir()
        output = parent / "release-v1"
        result = SimpleNamespace(
            participant_ids=np.asarray(["grp_" + "a" * 64], dtype=object),
            cohorts=np.asarray(["als"], dtype=object),
            labels=np.asarray([1], dtype=np.int64),
            fold_assignments=np.asarray([0], dtype=np.int64),
            task_scores=np.asarray([[0.7, 0.8, 0.9]], dtype=np.float64),
            participant_scores=np.asarray([0.8], dtype=np.float64),
            original_probabilities=np.asarray([[0.6, 0.7, 0.8]], dtype=np.float64),
            mirrored_probabilities=np.asarray([[0.8, 0.9, 1.0]], dtype=np.float64),
        )
        original_open = runner._open_secure_directory_fd

        def open_then_swap(path):
            descriptor = original_open(path)
            parent.rename(saved_parent)
            parent.symlink_to(attacker, target_is_directory=True)
            return descriptor

        runner._open_secure_directory_fd = open_then_swap
        try:
            runner._write_release_atomically(
                output, {"schema_version": "fixture"}, result
            )
        finally:
            runner._open_secure_directory_fd = original_open
        c.true((saved_parent / "release-v1" / "FINALIZATION.json").is_file(),
               "publication remains anchored to the checked parent inode")
        c.true(not (attacker / "release-v1").exists(),
               "attacker replacement path receives no private artifact")


def test_palsynet_named_input_is_rejected_before_read(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "PalsyNet-do-not-open" / "manifest.json"
        c.raises(lambda: runner._reject_palsynet_path(missing), ValueError)
        c.true(not missing.exists())


def test_runner_source_binds_authoritative_feature_api_and_no_unsafe_npz_loader(c: Check):
    source = (ROOT / "scripts" / "run_neuroface_action_capacity_v1.py").read_text()
    c.true("validate_neuroface_task_binding" in source)
    c.true("neuroface_action_capacity_feature_vector" in source)
    c.true("mirror_action_capacity_features" in source)
    c.true("load_dynamic_landmark_recording" not in source)
    c.true("np.load" not in source)
    c.true("PalsyNet" not in source, "runner does not name or import protected data")
    c.eq(set(runner._implementation_components_sha256()), {
        "scripts/run_neuroface_action_capacity_v1.py",
        "scripts/launch_neuroface_action_capacity_v1.py",
        "scripts/run_mirror_invariant_110d.py",
        "environment/neuroface_action_capacity_host_audit_ed25519_public.pem",
        "src/datasets/dynamic_landmark.py",
        "src/evaluation/neuroface_action_capacity_v1.py",
        "src/preprocessing/action_capacity_features_v1.py",
        "src/preprocessing/generalization_110d.py",
        "src/preprocessing/script_action_segmentation_v1.py",
        "src/preprocessing/trajectory_features.py",
        "src/datasets/patient_multistream.py",
        "src/preprocessing/clinical_landmarks.py",
        "src/training/neuroface_motion_pretrain_v1.py",
    })

    before = runner._implementation_components_sha256()
    original = runner._implementation_components_sha256
    runner._implementation_components_sha256 = lambda: {**before, "changed.py": "0" * 64}
    try:
        c.raises(lambda: runner._verify_implementation_unchanged(before), ValueError)
    finally:
        runner._implementation_components_sha256 = original


def test_dataset_builder_passes_the_same_authoritative_bytes_for_all_108_rows(c: Check):
    private_payload = b"exact-private-manifest-fixture"
    collection_payload = b"exact-collection-manifest-fixture"
    rows = []
    cache_payloads = {}
    for participant in range(36):
        cohort = ("als" if participant < 11 else
                  "healthy_control" if participant < 22 else "post_stroke")
        for task_index, task in enumerate(("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")):
            recording_id = f"rec_{participant * 3 + task_index:064x}"
            payload = f"cache-{recording_id}".encode("ascii")
            cache_payloads[recording_id] = payload
            rows.append({
                "recording_id": recording_id,
                "participant_id": f"grp_{participant:064x}",
                "video_sha256": f"{participant * 3 + task_index + 1000:064x}",
                "task": task,
                "cohort": cohort,
            })
    seen = {"bindings": 0, "features": 0, "mirrors": 0}
    originals = {
        name: getattr(runner, name) for name in (
            "PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256",
            "_primary_manifest_rows", "validate_neuroface_task_binding",
            "neuroface_action_capacity_feature_vector",
            "mirror_action_capacity_features",
        )
    }
    try:
        runner.PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256 = hashlib.sha256(
            collection_payload
        ).hexdigest()
        runner._primary_manifest_rows = lambda payload: (
            rows if payload is private_payload else (_ for _ in ()).throw(
                AssertionError("private bytes were copied or replaced")
            )
        )

        def binding(payload, *, recording_id, decoded_recording_sha256):
            c.true(payload is private_payload)
            row = next(item for item in rows if item["recording_id"] == recording_id)
            c.eq(decoded_recording_sha256, row["video_sha256"])
            seen["bindings"] += 1
            return SimpleNamespace(task_label=row["task"], recording_id=recording_id)

        def feature(bound, cache_payload, collection, *, decoded_recording_sha256):
            c.true(collection is collection_payload)
            c.true(cache_payload is cache_payloads[bound.recording_id])
            seen["features"] += 1
            index = int(bound.recording_id[4:], 16)
            return np.full(18, float(index), dtype=np.float64)

        def mirror(values):
            seen["mirrors"] += 1
            return np.asarray(values[::-1], dtype=np.float64)

        runner.validate_neuroface_task_binding = binding
        runner.neuroface_action_capacity_feature_vector = feature
        runner.mirror_action_capacity_features = mirror
        dataset, digest = runner._build_dataset_from_authoritative_bytes(
            private_payload, collection_payload,
            lambda recording_id: cache_payloads[recording_id],
        )
    finally:
        for name, value in originals.items():
            setattr(runner, name, value)
    c.eq(seen, {"bindings": 108, "features": 108, "mirrors": 108})
    c.eq(dataset.original_features.shape, (108, 18))
    c.eq(dataset.mirrored_features.shape, (108, 18))
    c.eq(len(digest), 64)


if __name__ == "__main__":
    run_all("test_run_neuroface_action_capacity_v1", dict(globals()))
