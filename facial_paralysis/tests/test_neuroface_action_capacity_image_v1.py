"""Minimal immutable image contract for the formal NeuroFace experiment."""
from __future__ import annotations

from pathlib import Path

from _testlib import Check, run_all


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "environment" / "neuroface_action_capacity_v1.Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def test_image_extends_the_frozen_dependency_layer(c: Check):
    c.true(DOCKERFILE.is_file(), "the v1.4 runtime Dockerfile exists")
    source = DOCKERFILE.read_text(encoding="utf-8")
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    c.eq(lines[0], "FROM facial-paralysis-neuroface:v1.3")
    c.true("WORKDIR /workspace/facial_paralysis" in lines)
    c.true("USER 1001:1001" in lines)
    c.true("CMD [\"python\", \"scripts/run_neuroface_action_capacity_v1.py\", \"--help\"]" in lines)


def test_image_copies_only_the_frozen_runtime_surface(c: Check):
    source = DOCKERFILE.read_text(encoding="utf-8")
    required = (
        "COPY src /workspace/facial_paralysis/src",
        "COPY scripts/run_neuroface_action_capacity_v1.py /workspace/facial_paralysis/scripts/run_neuroface_action_capacity_v1.py",
        "COPY scripts/run_mirror_invariant_110d.py /workspace/facial_paralysis/scripts/run_mirror_invariant_110d.py",
        "COPY scripts/launch_neuroface_action_capacity_v1.py /workspace/facial_paralysis/scripts/launch_neuroface_action_capacity_v1.py",
        "COPY environment/neuroface_h200_v1.lock /workspace/facial_paralysis/environment/neuroface_h200_v1.lock",
        "COPY environment/neuroface_action_capacity_host_audit_ed25519_public.pem /workspace/facial_paralysis/environment/neuroface_action_capacity_host_audit_ed25519_public.pem",
    )
    for statement in required:
        c.true(statement in source, f"runtime image includes {statement}")
    lowered = source.casefold()
    for forbidden in (
        "copy . ", "add ", "data/", "outputs/", "tests/", "private-ed25519",
        "palsynet", "volume ",
    ):
        c.true(forbidden not in lowered, f"runtime image excludes {forbidden}")


def test_build_context_is_closed_and_excludes_python_bytecode(c: Check):
    c.true(DOCKERIGNORE.is_file(), "the formal build context has an allowlist")
    patterns = {
        line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in (
        "**", "!src/", "!src/**", "!scripts/",
        "!scripts/run_neuroface_action_capacity_v1.py",
        "!scripts/run_mirror_invariant_110d.py",
        "!scripts/launch_neuroface_action_capacity_v1.py",
        "!environment/", "!environment/neuroface_h200_v1.lock",
        "!environment/neuroface_action_capacity_host_audit_ed25519_public.pem",
        "!environment/neuroface_action_capacity_v1.Dockerfile",
        "**/__pycache__/", "**/*.pyc", "**/*.pyo",
    ):
        c.true(required in patterns, f"closed context includes rule {required}")
    tracked_bytecode = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix in {".pyc", ".pyo"}
        and ".git" not in path.parts
    ]
    c.true(bool(tracked_bytecode),
           "the regression fixture exercises a worktree that contains bytecode")


if __name__ == "__main__":
    run_all("test_neuroface_action_capacity_image_v1", dict(globals()))
