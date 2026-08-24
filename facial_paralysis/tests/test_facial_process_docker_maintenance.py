"""Destructive-boundary tests for Facial Process Docker maintenance."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from scripts.maintain_facial_process_docker import (  # noqa: E402
    BUILD_CACHE_MAX_USED_SPACE,
    BUILD_CACHE_MIN_AGE,
    BUILD_CACHE_RESERVED_SPACE,
    BUILDKITD_CONFIG,
    IMAGE_RETENTION_SECONDS,
    OWNERSHIP_LABEL_KEY,
    OWNERSHIP_LABEL_VALUE,
    PROJECT_BUILDER,
    CommandRunner,
    DockerSnapshot,
    ImageRecord,
    MaintenanceLock,
    apply_plan,
    build_plan,
    cap_local_service_logs,
    capture_snapshot,
    initialize_project_builder,
    snapshot_sha256,
)


NOW = 2_000_000_000
OLD = NOW - IMAGE_RETENTION_SECONDS - 1


def _image(
    suffix: str,
    *,
    created: int = OLD,
    tags: tuple[str, ...] = (),
    owned: bool = True,
    referenced: bool = False,
    size: int = 100,
) -> ImageRecord:
    labels = (
        ((OWNERSHIP_LABEL_KEY, OWNERSHIP_LABEL_VALUE),)
        if owned
        else (("unrelated", "true"),)
    )
    return ImageRecord(
        image_id="sha256:" + suffix * 64,
        created_unix=created,
        size_bytes=size,
        repo_tags=tags,
        labels=labels,
        referenced_by_container=referenced,
    )


class FakeRunner(CommandRunner):
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def run(self, args: tuple[str, ...]) -> str:
        self.calls.append(args)
        if args[:2] == ("docker", "exec"):
            return """
[log]
format = "text"
level = "warn"
[worker]
[worker.oci]
gc = true
max-parallelism = 4
maxUsedSpace = "8GB"
minFreeSpace = "20GB"
reservedSpace = "2GB"
"""
        return ""


def test_plan_selects_only_old_owned_dangling_unused_images(c: Check):
    eligible = _image("a", size=700)
    records = (
        eligible,
        _image("b", owned=False),
        _image("c", created=NOW - 60),
        _image("d", referenced=True),
        _image("e", tags=("facial-process-shared-v9-web:old",)),
        _image("f", tags=("some-other-project:latest",)),
        _image("9", created=-1),
    )
    plan = build_plan(DockerSnapshot(records), now_unix=NOW, builder_available=True)
    c.eq(plan.image_ids, (eligible.image_id,))
    c.eq(plan.image_reclaimable_bytes, 700)
    c.true(plan.prune_project_builder)


def test_snapshot_digest_is_order_independent_and_detects_drift(c: Check):
    a, b = _image("a"), _image("b")
    first = DockerSnapshot((a, b))
    second = DockerSnapshot((b, a))
    c.eq(snapshot_sha256(first), snapshot_sha256(second))
    c.true(snapshot_sha256(first) != snapshot_sha256(DockerSnapshot((a, replace(b, size_bytes=101)))))


def test_apply_rejects_stale_inventory_before_any_mutation(c: Check):
    initial = DockerSnapshot((_image("a"),))
    plan = build_plan(initial, now_unix=NOW, builder_available=True)
    runner = FakeRunner()
    c.raises(
        lambda: apply_plan(
            plan,
            DockerSnapshot((_image("a", size=101),)),
            runner=runner,
            now_unix=NOW + 1,
            builder_driver="docker-container",
        ),
        ValueError,
    )
    c.eq(runner.calls, [])


def test_apply_uses_only_exact_project_scoped_delete_commands(c: Check):
    snapshot = DockerSnapshot((_image("a"),))
    plan = build_plan(snapshot, now_unix=NOW, builder_available=True)
    runner = FakeRunner()
    apply_plan(
        plan,
        snapshot,
        runner=runner,
        now_unix=NOW + 1,
        builder_driver="docker-container",
    )
    c.eq(
        runner.calls,
        [
            ("docker", "buildx", "inspect", PROJECT_BUILDER, "--bootstrap"),
            (
                "docker", "exec", f"buildx_buildkit_{PROJECT_BUILDER}0",
                "cat", "/etc/buildkit/buildkitd.toml",
            ),
            (
                "docker",
                "buildx",
                "prune",
                "--builder",
                PROJECT_BUILDER,
                "--filter",
                f"until={BUILD_CACHE_MIN_AGE}",
                "--max-used-space",
                BUILD_CACHE_MAX_USED_SPACE,
                "--reserved-space",
                BUILD_CACHE_RESERVED_SPACE,
                "--force",
            ),
            ("docker", "image", "rm", "sha256:" + "a" * 64),
        ],
    )
    flattened = " ".join(" ".join(call) for call in runner.calls)
    c.true("system prune" not in flattened)
    c.true("volume" not in flattened)
    c.true("--all" not in flattened)


def test_builder_preflight_failure_cannot_delete_an_image(c: Check):
    class FailingRunner(CommandRunner):
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []

        def run(self, args: tuple[str, ...]) -> str:
            self.calls.append(args)
            raise RuntimeError("builder unavailable")

    snapshot = DockerSnapshot((_image("a"),))
    plan = build_plan(snapshot, now_unix=NOW, builder_available=True)
    runner = FailingRunner()
    c.raises(
        lambda: apply_plan(
            plan,
            snapshot,
            runner=runner,
            now_unix=NOW + 1,
            builder_driver="docker-container",
        ),
        RuntimeError,
    )
    c.eq(runner.calls, [("docker", "buildx", "inspect", PROJECT_BUILDER, "--bootstrap")])


def test_builder_config_drift_cannot_prune_or_delete(c: Check):
    class DriftRunner(CommandRunner):
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []

        def run(self, args: tuple[str, ...]) -> str:
            self.calls.append(args)
            if args[:2] == ("docker", "exec"):
                return '[log]\nlevel = "info"\n'
            return ""

    snapshot = DockerSnapshot((_image("a"),))
    plan = build_plan(snapshot, now_unix=NOW, builder_available=True)
    runner = DriftRunner()
    c.raises(
        lambda: apply_plan(
            plan,
            snapshot,
            runner=runner,
            now_unix=NOW + 1,
            builder_driver="docker-container",
        ),
        ValueError,
    )
    c.eq(len(runner.calls), 2)
    c.true(all("prune" not in call and "rm" not in call for call in runner.calls))


def test_apply_rejects_wrong_or_missing_builder_driver(c: Check):
    snapshot = DockerSnapshot(())
    plan = build_plan(snapshot, now_unix=NOW, builder_available=True)
    for driver in ("docker", "kubernetes", ""):
        runner = FakeRunner()
        c.raises(
            lambda driver=driver: apply_plan(
                plan,
                snapshot,
                runner=runner,
                now_unix=NOW + 1,
                builder_driver=driver,
            ),
            ValueError,
        )
        c.eq(runner.calls, [])


def test_apply_rejects_expired_plan(c: Check):
    snapshot = DockerSnapshot((_image("a"),))
    plan = build_plan(snapshot, now_unix=NOW, builder_available=False)
    runner = FakeRunner()
    c.raises(
        lambda: apply_plan(
            plan,
            snapshot,
            runner=runner,
            now_unix=NOW + 901,
            builder_driver=None,
        ),
        ValueError,
    )
    c.eq(runner.calls, [])


def test_lock_fails_closed_under_concurrent_maintenance(c: Check):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "maintenance.lock"
        with MaintenanceLock(path):
            c.raises(lambda: MaintenanceLock(path).__enter__(), RuntimeError)


def test_docker_unavailable_fails_before_inventory_or_mutation(c: Check):
    class OfflineRunner(CommandRunner):
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []

        def run(self, args: tuple[str, ...]) -> str:
            self.calls.append(args)
            raise RuntimeError("Docker daemon unavailable")

    runner = OfflineRunner()
    c.raises(lambda: capture_snapshot(runner), RuntimeError)
    c.eq(runner.calls, [("docker", "info", "--format", "{{json .ServerVersion}}")])


def test_builder_initialization_is_named_isolated_and_not_global(c: Check):
    class BuilderRunner(CommandRunner):
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []
            self.inspections = 0

        def run(self, args: tuple[str, ...]) -> str:
            self.calls.append(args)
            if args[:3] == ("docker", "buildx", "inspect"):
                self.inspections += 1
                if self.inspections == 1:
                    raise RuntimeError("ERROR: no builder facial-process-v9-builder found")
                return "Name: facial-process-v9-builder\nDriver: docker-container\n"
            if args[:2] == ("docker", "exec"):
                return """
[log]
format = "text"
level = "warn"
[worker]
[worker.oci]
gc = true
max-parallelism = 4
maxUsedSpace = "8GB"
minFreeSpace = "20GB"
reservedSpace = "2GB"
"""
            return "facial-process-v9-builder"

    runner = BuilderRunner()
    initialize_project_builder(runner)
    c.eq(
        runner.calls,
        [
            ("docker", "buildx", "inspect", PROJECT_BUILDER),
            (
                "docker", "buildx", "create", "--name", PROJECT_BUILDER,
                "--driver", "docker-container", "--driver-opt", "default-load=true",
                "--buildkitd-config", str(BUILDKITD_CONFIG),
            ),
            ("docker", "buildx", "inspect", PROJECT_BUILDER),
            ("docker", "buildx", "inspect", PROJECT_BUILDER, "--bootstrap"),
            (
                "docker", "exec", f"buildx_buildkit_{PROJECT_BUILDER}0",
                "cat", "/etc/buildkit/buildkitd.toml",
            ),
        ],
    )


def test_log_cap_covers_all_three_files_and_rejects_links(c: Check):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        small = root / "service.log"
        large = root / "service.err.log"
        reconcile = root / "last-reconcile-error.log"
        small.write_bytes(b"ok")
        large.write_bytes(b"x" * (1_048_576 + 1))
        reconcile.symlink_to(large)
        c.raises(lambda: cap_local_service_logs(root), ValueError)
        c.eq(small.read_bytes(), b"ok")
        c.eq(large.stat().st_size, 1_048_577)
        reconcile.unlink()
        reconcile.write_bytes(b"y" * (1_048_576 + 1))
        changed = cap_local_service_logs(root)
        c.eq(changed, ("last-reconcile-error.log", "service.err.log"))
        c.eq(small.read_bytes(), b"ok")
        c.eq(large.stat().st_size, 0)
        c.eq(reconcile.stat().st_size, 0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        outside = root / "outside.log"
        outside.write_bytes(b"z" * (1_048_576 + 1))
        os_link = root / "service.err.log"
        os.link(outside, os_link)
        c.raises(lambda: cap_local_service_logs(root), ValueError)
        c.eq(outside.stat().st_size, 1_048_577)


def test_release_files_bind_ownership_logs_and_dedicated_builder(c: Check):
    dockerfiles = (
        ROOT / "environment/shared_v9_public_v1.Dockerfile",
        ROOT / "environment/faces_shared_v9_gateway_v1.Dockerfile",
        ROOT / "facial_paralysis_web/Dockerfile",
    )
    for path in dockerfiles:
        text = path.read_text(encoding="utf-8")
        c.true(f'{OWNERSHIP_LABEL_KEY}="{OWNERSHIP_LABEL_VALUE}"' in text, str(path))
    readme = (ROOT / "deploy/facial-process-shared-v9/README.md").read_text(encoding="utf-8")
    c.true(PROJECT_BUILDER in readme)
    c.true("maintain_facial_process_docker.py audit" in readme)
    c.true("docker system prune" in readme)
    c.true("must not" in readme)
    buildkitd = BUILDKITD_CONFIG.read_text(encoding="utf-8")
    c.true('level = "warn"' in buildkitd)
    c.true('reservedSpace = "2GB"' in buildkitd)
    c.true('maxUsedSpace = "8GB"' in buildkitd)
    c.true('minFreeSpace = "20GB"' in buildkitd)


if __name__ == "__main__":
    run_all("test_facial_process_docker_maintenance", dict(globals()))
