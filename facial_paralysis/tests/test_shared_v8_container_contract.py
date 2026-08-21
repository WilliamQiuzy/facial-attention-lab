from __future__ import annotations

from pathlib import Path
import tempfile

from _testlib import run_all

from scripts.accept_shared_v8_deployment_v1 import synthetic_requests
from scripts.launch_shared_v8_container_v1 import build_run_command
from src.deployment.shared_v8_service import MAX_REQUEST_BYTES


ROOT = Path(__file__).resolve().parents[1]


def test_docker_build_is_digest_pinned_closed_and_nonroot(c):
    dockerfile = (ROOT / "environment/shared_v8_deployment_v1.Dockerfile").read_text()
    ignored = (ROOT / ".dockerignore").read_text()
    lock = (ROOT / "environment/shared_v8_runtime_v1.lock").read_text().splitlines()
    c.true("pytorch/pytorch@sha256:c16f4c749e2d" in dockerfile)
    c.true("USER 1001:1001" in dockerfile)
    c.true("COPY ." not in dockerfile and "ADD " not in dockerfile)
    c.true("--no-deps" in dockerfile and "pip check" in dockerfile)
    c.true(ignored.startswith("# Closed runtime allowlist"))
    c.true("**/__pycache__/" in ignored and "**/*.pyc" in ignored)
    c.eq(len(lock), 13)
    c.true(all("==" in row and " " not in row for row in lock))


def test_launcher_enforces_readonly_localhost_nonroot_gpu_contract(c):
    with tempfile.TemporaryDirectory() as temporary:
        release = Path(temporary) / "release"
        release.mkdir()
        command = build_run_command(
            image_id="sha256:" + "a" * 64,
            release_root=release,
            port=18080,
            name="shared-v8-deployment-v1",
        )
        rendered = " ".join(command)
        for token in (
            "--gpus device=0", "--read-only", "--cap-drop=ALL",
            "no-new-privileges:true", "--user=1001:1001",
            "127.0.0.1:18080:8080", "dst=/model,readonly", "--pids-limit=256",
        ):
            c.true(token in rendered, token)
        c.eq(command[-1], "sha256:" + "a" * 64)
        c.raises(
            lambda: build_run_command(
                image_id="shared-v8:latest", release_root=release,
                port=18080, name="shared-v8-deployment-v1",
            ),
            ValueError,
        )


def test_acceptance_payloads_cover_all_clinical_protocols(c):
    payloads = synthetic_requests()
    c.eq(set(payloads), {
        "free_motion_four_window", "scripted_three_action", "cue_aligned_action"
    })
    c.true(all(type(value) is bytes and 0 < len(value) < MAX_REQUEST_BYTES
               for value in payloads.values()))


if __name__ == "__main__":
    run_all("test_shared_v8_container_contract", dict(globals()))
