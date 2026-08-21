from __future__ import annotations

from pathlib import Path
import json
import re

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]


def test_v9_image_is_closed_nonroot_and_contains_exact_public_weights(c):
    dockerfile = (ROOT / "environment/shared_v9_public_v1.Dockerfile").read_text()
    ignored = (ROOT / ".dockerignore").read_text()
    c.true("pytorch/pytorch@sha256:c16f4c749e2d" in dockerfile)
    c.true("SHARED_V9_DEVICE=cpu" in dockerfile)
    c.true("USER 1001:1001" in dockerfile)
    c.true("COPY ." not in dockerfile and "ADD " not in dockerfile)
    c.true("--no-deps" in dockerfile and "pip check" in dockerfile)
    for digest in (
        "7befb2853b89a11ebf904483b027098d042e36d12891d65d93ffc4766ad3fc96",
        "9f27e5d3535a472c09cb9cfd94cadd432d8373242b73a67e6d533e287d76760f",
        "b97ef723ee8a2fdb6c90d04c1f5c1adb0b090292a76313feddc46f3b9a68fdf7",
    ):
        c.true(digest in dockerfile)
    c.true("org.opencontainers.image.source" in dockerfile)
    c.true("!environment/shared_v9_public_v1.Dockerfile" in ignored)
    c.true("!scripts/serve_shared_v9.py" in ignored)
    c.true("!releases/shared-v9-research-v1/manifest.json" in ignored)
    c.true("!releases/shared-v9-research-v1/weights-seed0.npz" in ignored)


def test_cpu_default_and_optional_gpu_compose_are_hardened(c):
    cpu = (ROOT / "deploy/shared-v9/compose.yaml").read_text()
    gpu = (ROOT / "deploy/shared-v9/compose.gpu.yaml").read_text()
    c.true(
        "image: ghcr.io/williamqiuzy/facial-attention-lab-shared-v9@"
        "sha256:ec0e2b34e2233e159d555ab3761fe113f5b768562ba9d9d7bf7c2d7a27d42c95"
        in cpu
    )
    c.true(":latest" not in cpu and ":v9-public" not in cpu)
    for token in (
        "SHARED_V9_DEVICE: cpu", "read_only: true", 'user: "1001:1001"',
        "cap_drop:", "- ALL", "no-new-privileges:true",
        "127.0.0.1:18090:8080", "tmpfs:", "pids_limit: 256",
    ):
        c.true(token in cpu, token)
    c.true("capabilities: [gpu]" not in cpu)
    c.true("SHARED_V9_DEVICE: cuda" in gpu)
    c.true("capabilities: [gpu]" in gpu)


def test_public_quickstart_is_anonymous_and_one_command(c):
    quickstart = (ROOT / "deploy/shared-v9/README.md").read_text().lower()
    for phrase in (
        "public", "docker compose pull", "docker compose up -d", "/readyz",
        "compose.gpu.yaml", "not clinically validated", "preprocessed",
    ):
        c.true(phrase in quickstart, phrase)
    c.true("docker login" not in quickstart)


def test_public_oci_release_binds_image_and_h200_acceptance(c):
    release = json.loads(
        (ROOT / "releases/shared-v9-research-v1/oci_manifest.json").read_bytes()
    )
    c.eq(release["schema_version"], "shared_v9_public_oci_v1")
    c.eq(release["visibility"], "public")
    c.eq(
        release["image_digest"],
        "sha256:ec0e2b34e2233e159d555ab3761fe113f5b768562ba9d9d7bf7c2d7a27d42c95",
    )
    c.eq(
        release["image_id"],
        "sha256:86f6018e1049c6c599d15f31ff52059667a4cecbfe2d041c7b78499fc421a79b",
    )
    c.eq(release["source_commit"], "452cb9be173ec472743acc1c617368f89b3cd97c")
    c.true(release["acceptance"]["cpu"] is True)
    c.true(release["acceptance"]["gpu_h200"] is True)
    c.true(release["acceptance"]["cpu_gpu_maximum_absolute_difference"] < 1e-5)
    c.true(release["clinical_validation"] is False)
    registry = json.loads((ROOT / "docs/model_registry.json").read_bytes())
    deployment = registry["deployment"]
    c.eq(deployment["name"], release["model_id"])
    c.eq(deployment["version"], "shared-v9-public-oci-v1")
    c.eq(deployment["image_digest"], release["image_digest"])
    c.eq(
        deployment["oci_manifest_sha256"],
        "a13d3d2abc3365c5cb7b6afd23a4e64962a0ad7d4400f7eb366aadb073650ddd",
    )


if __name__ == "__main__":
    run_all("test_shared_v9_public_container", dict(globals()))
