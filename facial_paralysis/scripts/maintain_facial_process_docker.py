#!/usr/bin/env python3
"""Fail-closed, project-scoped Docker storage maintenance for Facial Process.

There is deliberately no global Docker prune or volume-delete path. Images
are eligible only when dangling, old, unused by every Docker container, and
marked with this release's exact ownership label. Build cache cleanup is
allowed only on the dedicated docker-container builder.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence


OWNERSHIP_LABEL_KEY = "io.github.williamqiu.facial-process.storage-scope"
OWNERSHIP_LABEL_VALUE = "shared-v9"
PROJECT_BUILDER = "facial-process-v9-builder"
BUILDKITD_CONFIG = Path(__file__).resolve().parents[1] / "deploy/facial-process-shared-v9/buildkitd.toml"
BUILDER_CONTAINER = f"buildx_buildkit_{PROJECT_BUILDER}0"
IMAGE_RETENTION_SECONDS = 7 * 24 * 60 * 60
PLAN_MAX_AGE_SECONDS = 15 * 60
BUILD_CACHE_MIN_AGE = "168h"
BUILD_CACHE_MAX_USED_SPACE = "8GB"
BUILD_CACHE_RESERVED_SPACE = "2GB"
LOG_CAP_BYTES = 1_048_576
PLAN_SCHEMA = "facial-process-docker-maintenance-plan/v1"
LOG_NAMES = ("last-reconcile-error.log", "service.err.log", "service.log")
EXPECTED_RUNTIME_CONFIG_LINES = frozenset(
    {
        "[log]",
        'format = "text"',
        'level = "warn"',
        "[worker]",
        "[worker.oci]",
        "gc = true",
        "max-parallelism = 4",
        'maxUsedSpace = "8GB"',
        'minFreeSpace = "20GB"',
        'reservedSpace = "2GB"',
    }
)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    created_unix: int
    size_bytes: int
    repo_tags: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]
    referenced_by_container: bool


@dataclass(frozen=True)
class DockerSnapshot:
    images: tuple[ImageRecord, ...]


@dataclass(frozen=True)
class CleanupPlan:
    schema_version: str
    created_unix: int
    inventory_sha256: str
    retention_seconds: int
    image_ids: tuple[str, ...]
    image_reclaimable_bytes: int
    project_builder: str
    prune_project_builder: bool


class CommandRunner:
    """Small injectable command boundary used by tests and the CLI."""

    def run(self, args: tuple[str, ...]) -> str:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1:] or ["command failed"]
            raise RuntimeError(f"Docker command failed: {args[:3]!r}: {detail[0]}")
        return completed.stdout


class MaintenanceLock:
    """Cross-process lock; the stable lock file is never unlinked."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> "MaintenanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            raise RuntimeError("Facial Process Docker maintenance is already running") from exc
        self._fd = fd
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _record_json(record: ImageRecord) -> dict[str, object]:
    return {
        "image_id": record.image_id,
        "created_unix": record.created_unix,
        "size_bytes": record.size_bytes,
        "repo_tags": list(sorted(record.repo_tags)),
        "labels": [[key, value] for key, value in sorted(record.labels)],
        "referenced_by_container": record.referenced_by_container,
    }


def snapshot_sha256(snapshot: DockerSnapshot) -> str:
    records = [_record_json(record) for record in sorted(snapshot.images, key=lambda item: item.image_id)]
    return hashlib.sha256(_canonical_json({"images": records})).hexdigest()


def _is_cleanup_candidate(record: ImageRecord, *, now_unix: int) -> bool:
    labels = dict(record.labels)
    digest = record.image_id.removeprefix("sha256:")
    return bool(
        record.image_id.startswith("sha256:")
        and len(record.image_id) == 71
        and all(character in "0123456789abcdef" for character in digest)
        and not record.repo_tags
        and labels.get(OWNERSHIP_LABEL_KEY) == OWNERSHIP_LABEL_VALUE
        and not record.referenced_by_container
        and record.created_unix >= 0
        and record.created_unix <= now_unix - IMAGE_RETENTION_SECONDS
    )


def build_plan(
    snapshot: DockerSnapshot,
    *,
    now_unix: int,
    builder_available: bool,
) -> CleanupPlan:
    candidates = tuple(
        sorted(
            (record for record in snapshot.images if _is_cleanup_candidate(record, now_unix=now_unix)),
            key=lambda item: item.image_id,
        )
    )
    return CleanupPlan(
        schema_version=PLAN_SCHEMA,
        created_unix=now_unix,
        inventory_sha256=snapshot_sha256(snapshot),
        retention_seconds=IMAGE_RETENTION_SECONDS,
        image_ids=tuple(record.image_id for record in candidates),
        image_reclaimable_bytes=sum(record.size_bytes for record in candidates),
        project_builder=PROJECT_BUILDER,
        prune_project_builder=bool(builder_available),
    )


def apply_plan(
    plan: CleanupPlan,
    current_snapshot: DockerSnapshot,
    *,
    runner: CommandRunner,
    now_unix: int,
    builder_driver: str | None,
) -> None:
    if plan.schema_version != PLAN_SCHEMA:
        raise ValueError("unsupported maintenance plan schema")
    age = now_unix - plan.created_unix
    if age < 0 or age > PLAN_MAX_AGE_SECONDS:
        raise ValueError("maintenance plan is expired or from the future")
    if plan.retention_seconds != IMAGE_RETENTION_SECONDS:
        raise ValueError("maintenance retention policy drift")
    if plan.project_builder != PROJECT_BUILDER:
        raise ValueError("maintenance builder drift")
    if snapshot_sha256(current_snapshot) != plan.inventory_sha256:
        raise ValueError("Docker inventory changed after audit; generate a new plan")
    recomputed = build_plan(
        current_snapshot,
        now_unix=plan.created_unix,
        builder_available=plan.prune_project_builder,
    )
    if recomputed != plan:
        raise ValueError("maintenance plan does not match the audited inventory")
    if plan.prune_project_builder and builder_driver != "docker-container":
        raise ValueError("project cache cleanup requires the dedicated docker-container builder")
    if not plan.prune_project_builder and builder_driver is not None:
        raise ValueError("unexpected builder appeared after audit; generate a new plan")

    if plan.prune_project_builder:
        # Bootstrap is a preflight: no image deletion is allowed if the
        # isolated cache backend is unavailable.
        runner.run(("docker", "buildx", "inspect", PROJECT_BUILDER, "--bootstrap"))
        validate_project_builder_config(runner)
        runner.run(
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
            )
        )
    for image_id in plan.image_ids:
        runner.run(("docker", "image", "rm", image_id))


def _parse_created_unix(value: str) -> int:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if "." in raw:
        prefix, suffix = raw.split(".", 1)
        offset_index = next((idx for idx, char in enumerate(suffix) if char in "+-"), len(suffix))
        fraction, offset = suffix[:offset_index], suffix[offset_index:]
        raw = f"{prefix}.{fraction[:6].ljust(6, '0')}{offset}"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def capture_snapshot(runner: CommandRunner) -> DockerSnapshot:
    runner.run(("docker", "info", "--format", "{{json .ServerVersion}}"))
    image_ids = tuple(sorted(set(runner.run(("docker", "image", "ls", "-aq", "--no-trunc")).split())))
    container_ids = tuple(sorted(set(runner.run(("docker", "container", "ls", "-aq", "--no-trunc")).split())))
    referenced: set[str] = set()
    if container_ids:
        containers = json.loads(runner.run(("docker", "container", "inspect", *container_ids)))
        referenced = {str(container.get("Image", "")) for container in containers}
    raw_images: Sequence[Mapping[str, object]] = ()
    if image_ids:
        raw_images = json.loads(runner.run(("docker", "image", "inspect", *image_ids)))
    records: list[ImageRecord] = []
    for raw in raw_images:
        config = raw.get("Config") or {}
        if not isinstance(config, Mapping):
            raise ValueError("Docker image Config must be a mapping")
        raw_labels = config.get("Labels") or {}
        if not isinstance(raw_labels, Mapping):
            raise ValueError("Docker image Labels must be a mapping")
        image_id = str(raw.get("Id", ""))
        tags = raw.get("RepoTags") or []
        if not isinstance(tags, list):
            raise ValueError("Docker image RepoTags must be a list")
        records.append(
            ImageRecord(
                image_id=image_id,
                created_unix=(
                    -1
                    if raw.get("Created") is None
                    else _parse_created_unix(str(raw.get("Created")))
                ),
                size_bytes=int(raw.get("Size", 0)),
                repo_tags=tuple(sorted(str(tag) for tag in tags)),
                labels=tuple(sorted((str(key), str(value)) for key, value in raw_labels.items())),
                referenced_by_container=image_id in referenced,
            )
        )
    if len(records) != len(image_ids):
        raise ValueError("Docker image inventory count changed during inspection")
    return DockerSnapshot(tuple(sorted(records, key=lambda item: item.image_id)))


def probe_project_builder(runner: CommandRunner) -> str | None:
    try:
        output = runner.run(("docker", "buildx", "inspect", PROJECT_BUILDER))
    except RuntimeError as exc:
        if "no builder" in str(exc).lower():
            return None
        raise
    for line in output.splitlines():
        if line.startswith("Driver:"):
            return line.partition(":")[2].strip()
    raise ValueError("could not determine project builder driver")


def initialize_project_builder(runner: CommandRunner) -> None:
    driver = probe_project_builder(runner)
    if driver is None:
        runner.run(
            (
                "docker",
                "buildx",
                "create",
                "--name",
                PROJECT_BUILDER,
                "--driver",
                "docker-container",
                "--driver-opt",
                "default-load=true",
                "--buildkitd-config",
                str(BUILDKITD_CONFIG),
            )
        )
        driver = probe_project_builder(runner)
    if driver != "docker-container":
        raise ValueError("project builder exists with the wrong driver")
    runner.run(("docker", "buildx", "inspect", PROJECT_BUILDER, "--bootstrap"))
    validate_project_builder_config(runner)


def validate_project_builder_config(runner: CommandRunner) -> None:
    output = runner.run(
        ("docker", "exec", BUILDER_CONTAINER, "cat", "/etc/buildkit/buildkitd.toml")
    )
    lines = tuple(line.strip() for line in output.splitlines() if line.strip())
    if len(lines) != len(set(lines)) or frozenset(lines) != EXPECTED_RUNTIME_CONFIG_LINES:
        raise ValueError("dedicated builder is not running the frozen log/GC policy")


def cap_local_service_logs(log_dir: Path) -> tuple[str, ...]:
    root_stat = log_dir.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError("service log root must be a real directory")
    entries: list[tuple[Path, os.stat_result]] = []
    for name in LOG_NAMES:
        path = log_dir / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"unsafe service log entry: {name}")
        entries.append((path, metadata))

    changed: list[str] = []
    for path, expected in entries:
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            actual = os.fstat(fd)
            if (actual.st_dev, actual.st_ino, actual.st_nlink) != (expected.st_dev, expected.st_ino, 1):
                raise ValueError(f"service log changed during maintenance: {path.name}")
            if actual.st_size > LOG_CAP_BYTES:
                os.ftruncate(fd, 0)
                os.fsync(fd)
                changed.append(path.name)
        finally:
            os.close(fd)
    return tuple(sorted(changed))


def _plan_payload(plan: CleanupPlan) -> dict[str, object]:
    payload = asdict(plan)
    payload["image_ids"] = list(plan.image_ids)
    return payload


def _write_plan(path: Path, plan: CleanupPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = _canonical_json(_plan_payload(plan))
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_plan(path: Path) -> CleanupPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "created_unix", "inventory_sha256", "retention_seconds",
        "image_ids", "image_reclaimable_bytes", "project_builder", "prune_project_builder",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("maintenance plan has an invalid schema")
    image_ids = raw["image_ids"]
    scalar_types = {
        "schema_version": str,
        "created_unix": int,
        "inventory_sha256": str,
        "retention_seconds": int,
        "image_reclaimable_bytes": int,
        "project_builder": str,
        "prune_project_builder": bool,
    }
    if any(type(raw[name]) is not expected_type for name, expected_type in scalar_types.items()):
        raise ValueError("maintenance plan field types are invalid")
    if not isinstance(image_ids, list) or not all(type(value) is str for value in image_ids):
        raise ValueError("maintenance image IDs must be a list of strings")
    if image_ids != sorted(set(image_ids)):
        raise ValueError("maintenance image IDs must be unique and sorted")
    if any(
        not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
        for value in image_ids
    ):
        raise ValueError("maintenance image ID is invalid")
    inventory_sha = raw["inventory_sha256"]
    if len(inventory_sha) != 64 or any(character not in "0123456789abcdef" for character in inventory_sha):
        raise ValueError("maintenance inventory digest is invalid")
    if raw["created_unix"] < 0 or raw["image_reclaimable_bytes"] < 0:
        raise ValueError("maintenance plan numeric fields must be nonnegative")
    return CleanupPlan(
        schema_version=raw["schema_version"],
        created_unix=raw["created_unix"],
        inventory_sha256=raw["inventory_sha256"],
        retention_seconds=raw["retention_seconds"],
        image_ids=tuple(image_ids),
        image_reclaimable_bytes=raw["image_reclaimable_bytes"],
        project_builder=raw["project_builder"],
        prune_project_builder=raw["prune_project_builder"],
    )


def _summary(plan: CleanupPlan, *, mode: str) -> dict[str, object]:
    return {
        "schema_version": "facial-process-docker-maintenance-summary/v1",
        "mode": mode,
        "inventory_sha256": plan.inventory_sha256,
        "eligible_project_images": len(plan.image_ids),
        "eligible_project_image_bytes": plan.image_reclaimable_bytes,
        "project_builder": plan.project_builder,
        "project_builder_cache_prune": plan.prune_project_builder,
        "global_image_or_cache_prune": False,
        "volume_prune": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="write a short-lived, non-mutating cleanup plan")
    audit.add_argument("--plan", type=Path, required=True)
    apply = subparsers.add_parser("apply", help="apply an unchanged plan under the project lock")
    apply.add_argument("--plan", type=Path, required=True)
    subparsers.add_parser("init-builder", help="create/validate the dedicated project builder")
    logs = subparsers.add_parser("cap-logs", help="cap the three local watchdog log files")
    logs.add_argument("--log-dir", type=Path, default=Path.home() / "Library/Logs/FacialProcessV9")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runner = CommandRunner()
    lock_path = Path.home() / ".facial-process-v9/docker-maintenance.lock"
    try:
        if args.command == "init-builder":
            with MaintenanceLock(lock_path):
                initialize_project_builder(runner)
            print(json.dumps({"project_builder": PROJECT_BUILDER, "driver": "docker-container"}))
            return 0
        if args.command == "cap-logs":
            changed = cap_local_service_logs(args.log_dir)
            print(json.dumps({"capped_logs": list(changed), "cap_bytes": LOG_CAP_BYTES}))
            return 0
        if args.command == "audit":
            with MaintenanceLock(lock_path):
                snapshot = capture_snapshot(runner)
                driver = probe_project_builder(runner)
                if driver not in (None, "docker-container"):
                    raise ValueError("project builder exists with the wrong driver")
                plan = build_plan(snapshot, now_unix=int(time.time()), builder_available=driver == "docker-container")
                _write_plan(args.plan, plan)
            print(json.dumps(_summary(plan, mode="audit"), sort_keys=True))
            return 0
        if args.command == "apply":
            with MaintenanceLock(lock_path):
                plan = _load_plan(args.plan)
                snapshot = capture_snapshot(runner)
                driver = probe_project_builder(runner)
                apply_plan(plan, snapshot, runner=runner, now_unix=int(time.time()), builder_driver=driver)
            print(json.dumps(_summary(plan, mode="apply"), sort_keys=True))
            return 0
        raise AssertionError("unreachable")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"maintenance refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
