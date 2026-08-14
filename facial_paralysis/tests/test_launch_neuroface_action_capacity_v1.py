"""Host-side signed Docker attestation and launch contracts."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
import scripts.launch_neuroface_action_capacity_v1 as launcher  # noqa: E402


def _inspect(source_input: str, source_output: str) -> dict[str, object]:
    return {
        "Id": "d" * 64,
        "Image": "sha256:" + "a" * 64,
        "Config": {"User": "1001:1001"},
        "Mounts": [
            {
                "Type": "bind", "Source": source_output,
                "Destination": "/neuroface-output", "Mode": "",
                "RW": True, "Propagation": "rprivate",
            },
            {
                "Type": "bind", "Source": source_input,
                "Destination": "/neuroface-input", "Mode": "",
                "RW": False, "Propagation": "rprivate",
            },
        ],
    }


def test_mount_projection_is_closed_exact_and_committed(c: Check):
    input_source = os.fspath(launcher.HOST_INPUT_RELEASE_ROOT)
    output_source = os.fspath(launcher.HOST_OUTPUT_RELEASE_ROOT)
    inspect = _inspect(input_source, output_source)
    mounts, commitment = launcher._validated_mount_projection(inspect)
    c.eq([mount["destination"] for mount in mounts], [
        "/neuroface-input", "/neuroface-output",
    ])
    canonical = (json.dumps(
        mounts, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("ascii")
    c.eq(commitment, hashlib.sha256(canonical).hexdigest())
    inspect["Mounts"].append({
        "Type": "bind", "Source": "/protected", "Destination": "/extra",
        "Mode": "ro", "RW": False, "Propagation": "rprivate",
    })
    c.raises(lambda: launcher._validated_mount_projection(inspect), ValueError)


def test_tree_commitment_rejects_symlinks_hardlinks_and_hidden_extras(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        (root / "cache").mkdir(mode=0o700)
        member = root / "cache" / "member.npz"
        member.write_bytes(b"exact cache")
        os.chmod(member, 0o600)
        first = launcher._release_tree_commitment(root)
        c.eq(first, launcher._release_tree_commitment(root))
        symlink = root / "cache" / "link.npz"
        symlink.symlink_to(member)
        c.raises(lambda: launcher._release_tree_commitment(root), ValueError)
        symlink.unlink()
        hardlink = root / "cache" / "hard.npz"
        os.link(member, hardlink)
        c.raises(lambda: launcher._release_tree_commitment(root), ValueError)


def test_ed25519_signer_uses_owner_private_files_and_openssl_rawin(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        private = root / "private.pem"
        public = root / "public.pem"
        subprocess.run([
            "openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private),
        ], check=True, capture_output=True)
        os.chmod(private, 0o600)
        subprocess.run([
            "openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public),
        ], check=True, capture_output=True)
        message = b'{"signed":"canonical"}\n'
        signature = launcher._sign_payload(message, private_key_path=private)
        c.eq(len(signature), 64)
        message_path = root / "verify-message"
        signature_path = root / "verify-signature"
        message_path.write_bytes(message)
        signature_path.write_bytes(signature)
        verified = subprocess.run([
            "openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(public),
            "-rawin", "-in", str(message_path), "-sigfile", str(signature_path),
        ], capture_output=True)
        c.eq(verified.returncode, 0)


def test_launch_create_inspect_sign_start_order_and_no_third_bind(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        input_root = root / "input"
        output_root = root / "output"
        (input_root / "attestation").mkdir(parents=True, mode=0o700)
        output_root.mkdir(mode=0o700)
        os.chmod(input_root, 0o700)
        os.chmod(output_root, 0o700)
        private_key = root / "private.pem"
        private_key.write_bytes(b"fixture-private-key")
        os.chmod(private_key, 0o600)
        image_commitment = root / "image-id"
        image_commitment.write_bytes(("sha256:" + "a" * 64 + "\n").encode("ascii"))
        os.chmod(image_commitment, 0o600)
        events = []
        original = {
            "HOST_INPUT_RELEASE_ROOT": launcher.HOST_INPUT_RELEASE_ROOT,
            "HOST_OUTPUT_RELEASE_ROOT": launcher.HOST_OUTPUT_RELEASE_ROOT,
            "HOST_PRIVATE_KEY_PATH": launcher.HOST_PRIVATE_KEY_PATH,
            "HOST_IMAGE_ID_COMMITMENT_PATH": launcher.HOST_IMAGE_ID_COMMITMENT_PATH,
            "_validate_frozen_input_release": launcher._validate_frozen_input_release,
            "_release_tree_commitment": launcher._release_tree_commitment,
            "_validate_formal_output_release": launcher._validate_formal_output_release,
            "_docker": launcher._docker,
            "_sign_payload": launcher._sign_payload,
            "_verify_h200_host": launcher._verify_h200_host,
        }
        inspect = _inspect(os.fspath(input_root), os.fspath(output_root))

        def docker(args, *, timeout):
            events.append(("docker", tuple(args)))
            if args[:2] == ["container", "create"]:
                return SimpleNamespace(stdout="d" * 64 + "\n")
            if args[:2] == ["container", "inspect"]:
                return SimpleNamespace(stdout=json.dumps([inspect]))
            if args[:2] == ["container", "start"]:
                c.true((input_root / "attestation" / "host_attestation.json").is_file())
                return SimpleNamespace(stdout=json.dumps({
                    "schema_version": "neuroface_action_capacity_receipt_v1",
                    "report_sha256": "e" * 64,
                    "bootstrap_repeats": 5000,
                    "protected_data_accesses": 0,
                }) + "\n")
            raise AssertionError(args)

        try:
            launcher.HOST_INPUT_RELEASE_ROOT = input_root
            launcher.HOST_OUTPUT_RELEASE_ROOT = output_root
            launcher.HOST_PRIVATE_KEY_PATH = private_key
            launcher.HOST_IMAGE_ID_COMMITMENT_PATH = image_commitment
            launcher._validate_frozen_input_release = lambda path: "c" * 64
            launcher._release_tree_commitment = lambda path, **kwargs: (
                hashlib.sha256(b"[]\n").hexdigest()
                if path == output_root else "c" * 64
            )
            launcher._verify_h200_host = lambda: events.append(("h200",))
            launcher._validate_formal_output_release = lambda *args, **kwargs: None
            launcher._docker = docker
            launcher._sign_payload = lambda payload, *, private_key_path: (
                events.append(("sign", payload)) or b"s" * 64
            )
            receipt = launcher._launch_once()
        finally:
            for name, value in original.items():
                setattr(launcher, name, value)
        c.eq(json.loads(receipt)["schema_version"],
             "neuroface_action_capacity_receipt_v1")
        labels = [event[0] for event in events]
        c.true(labels.index("sign") > next(
            index for index, event in enumerate(events)
            if event[0] == "docker" and event[1][:2] == ("container", "inspect")
        ))
        c.true(labels.index("sign") < next(
            index for index, event in enumerate(events)
            if event[0] == "docker" and event[1][:2] == ("container", "start")
        ))
        create = next(event[1] for event in events
                      if event[0] == "docker" and event[1][:2] == ("container", "create"))
        c.eq(create.count("--mount"), 2)
        c.true("--user" in create)
        c.eq(create[create.index("--user") + 1], "1001:1001")
        c.true("--tmpfs" not in create,
               "the formal container has exactly the two signed mounts")
        c.true(os.fspath(launcher.HOST_PRIVATE_KEY_PATH) not in create,
               "host private key is never mounted into the container")


def test_formal_constants_and_public_key_are_frozen(c: Check):
    c.eq(launcher.CONTAINER_IMAGE, "facial-paralysis-neuroface:v1.4")
    c.eq(launcher.CONTAINER_NAME, "neuroface-action-capacity-v1")
    c.eq(launcher.HOST_PRIVATE_KEY_PATH, Path(
        "/home/ssh-ziyue/.config/facial-paralysis/"
        "action-capacity-attestation-v1/private-ed25519.pem"
    ))
    source = (ROOT / "scripts" / "launch_neuroface_action_capacity_v1.py").read_text()
    c.true("private-ed25519.pem" in source)
    c.true("read_bytes()" not in source,
           "the launcher never reads private key material into Python")


def test_image_commitment_receipt_and_failed_create_cleanup_are_closed(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        image_commitment = root / "image-id"
        image_commitment.write_bytes(("sha256:" + "a" * 64 + "\n").encode("ascii"))
        os.chmod(image_commitment, 0o600)
        c.eq(launcher._read_image_id_commitment(image_commitment),
             "sha256:" + "a" * 64)
        image_commitment.write_bytes(("sha256:" + "f" * 64 + "\n").encode("ascii"))
        c.eq(launcher._read_image_id_commitment(image_commitment),
             "sha256:" + "f" * 64)
        valid_receipt = json.dumps({
            "schema_version": "neuroface_action_capacity_receipt_v1",
            "report_sha256": "e" * 64,
            "bootstrap_repeats": 5000,
            "protected_data_accesses": 0,
        })
        c.eq(launcher._validated_receipt(valid_receipt)["report_sha256"], "e" * 64)
        c.raises(lambda: launcher._validated_receipt(json.dumps({
            "schema_version": "neuroface_action_capacity_receipt_v1",
        })), ValueError)

        input_root = root / "input"
        output_root = root / "output"
        (input_root / "attestation").mkdir(parents=True, mode=0o700)
        output_root.mkdir(mode=0o700)
        os.chmod(input_root, 0o700)
        os.chmod(output_root, 0o700)
        private_key = root / "private.pem"
        private_key.write_bytes(b"fixture")
        os.chmod(private_key, 0o600)
        events = []
        originals = {
            name: getattr(launcher, name) for name in (
                "HOST_INPUT_RELEASE_ROOT", "HOST_OUTPUT_RELEASE_ROOT",
                "HOST_PRIVATE_KEY_PATH", "HOST_IMAGE_ID_COMMITMENT_PATH",
                "_validate_frozen_input_release", "_release_tree_commitment",
                "_verify_h200_host", "_docker",
            )
        }

        def failing_docker(args, *, timeout):
            events.append(tuple(args))
            if args[:2] == ["container", "create"]:
                return SimpleNamespace(stdout="d" * 64 + "\n")
            if args[:2] == ["container", "inspect"]:
                arbitrary = _inspect(os.fspath(input_root), os.fspath(output_root))
                arbitrary["Image"] = "sha256:" + "a" * 64
                return SimpleNamespace(stdout=json.dumps([arbitrary]))
            if args[:2] == ["container", "rm"]:
                return SimpleNamespace(stdout="")
            raise AssertionError(args)

        try:
            launcher.HOST_INPUT_RELEASE_ROOT = input_root
            launcher.HOST_OUTPUT_RELEASE_ROOT = output_root
            launcher.HOST_PRIVATE_KEY_PATH = private_key
            launcher.HOST_IMAGE_ID_COMMITMENT_PATH = image_commitment
            launcher._validate_frozen_input_release = lambda path: "c" * 64
            launcher._release_tree_commitment = lambda path, **kwargs: (
                hashlib.sha256(b"[]\n").hexdigest()
            )
            launcher._verify_h200_host = lambda: None
            launcher._docker = failing_docker
            c.raises(launcher._launch_once, ValueError)
        finally:
            for name, value in originals.items():
                setattr(launcher, name, value)
        c.true(any(event[:2] == ("container", "rm") for event in events),
               "a created container is removed after pre-start failure")
        c.true(not (input_root / "attestation" / "host_attestation.json").exists(),
               "pre-start failure leaves no partial attestation")


def test_attestation_write_and_start_failure_leave_no_partial_state(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        attestation = root / "attestation"
        attestation.mkdir(mode=0o700)
        directory_fd = os.open(attestation, os.O_RDONLY | os.O_DIRECTORY)
        original_write = launcher.os.write
        wrote_once = False

        def interrupted_write(descriptor, payload):
            nonlocal wrote_once
            if not wrote_once:
                wrote_once = True
                original_write(descriptor, payload[:1])
            raise OSError("injected interrupted host write")

        launcher.os.write = interrupted_write
        try:
            c.raises(lambda: launcher._write_attestation_at_no_overwrite(
                directory_fd, "host_attestation.json", b"signed fixture\n"
            ), OSError)
        finally:
            launcher.os.write = original_write
            os.close(directory_fd)
        c.eq(list(attestation.iterdir()), [],
             "an interrupted publication leaves neither final nor temporary bytes")

        input_root = root / "input"
        output_root = root / "output"
        (input_root / "attestation").mkdir(parents=True, mode=0o700)
        output_root.mkdir(mode=0o700)
        os.chmod(input_root, 0o700)
        os.chmod(output_root, 0o700)
        image_commitment = root / "image-id"
        image_commitment.write_bytes(("sha256:" + "a" * 64 + "\n").encode("ascii"))
        os.chmod(image_commitment, 0o600)
        private_key = root / "private.pem"
        private_key.write_bytes(b"fixture")
        os.chmod(private_key, 0o600)
        events = []
        originals = {
            name: getattr(launcher, name) for name in (
                "HOST_INPUT_RELEASE_ROOT", "HOST_OUTPUT_RELEASE_ROOT",
                "HOST_PRIVATE_KEY_PATH", "HOST_IMAGE_ID_COMMITMENT_PATH",
                "_validate_frozen_input_release", "_release_tree_commitment",
                "_verify_h200_host", "_sign_payload", "_docker",
            )
        }

        def docker(args, *, timeout):
            events.append(tuple(args))
            if args[:2] == ["container", "create"]:
                return SimpleNamespace(stdout="d" * 64 + "\n")
            if args[:2] == ["container", "inspect"]:
                return SimpleNamespace(stdout=json.dumps([
                    _inspect(os.fspath(input_root), os.fspath(output_root))
                ]))
            if args[:2] == ["container", "start"]:
                raise RuntimeError("injected start failure")
            if args[:2] == ["container", "rm"]:
                return SimpleNamespace(stdout="")
            raise AssertionError(args)

        try:
            launcher.HOST_INPUT_RELEASE_ROOT = input_root
            launcher.HOST_OUTPUT_RELEASE_ROOT = output_root
            launcher.HOST_PRIVATE_KEY_PATH = private_key
            launcher.HOST_IMAGE_ID_COMMITMENT_PATH = image_commitment
            launcher._validate_frozen_input_release = lambda path: "c" * 64
            launcher._release_tree_commitment = lambda path, **kwargs: (
                hashlib.sha256(b"[]\n").hexdigest()
            )
            launcher._verify_h200_host = lambda: None
            launcher._sign_payload = lambda payload, **kwargs: b"s" * 64
            launcher._docker = docker
            c.raises(launcher._launch_once, RuntimeError)
        finally:
            for name, value in originals.items():
                setattr(launcher, name, value)
        c.true(any(event[:2] == ("container", "rm") for event in events),
               "start failure force-removes its created container")
        c.eq(list((input_root / "attestation").iterdir()), [],
             "start failure removes the complete attestation created by this attempt")


if __name__ == "__main__":
    run_all("test_launch_neuroface_action_capacity_v1", dict(globals()))
