"""Receipt-bound two-stage dynamic-landmark pretraining runner."""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import stat
import statistics
import sys
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINING_ROOT = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "pretraining"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_EXACT_RESULT_TREE_ENTRIES = 64
_MAX_EXACT_RESULT_TREE_DEPTH = 4
_MAX_EXACT_RESULT_TREE_REGULAR_BYTES = 128 * 1024 * 1024


def _authorization_factories(
    args: argparse.Namespace,
):
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    return inputs_cli._authorization_factories(args)


def _producer_sha256() -> str:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    return inputs_cli._producer_sha256()


def _privacy_forbidden(
    args: argparse.Namespace,
    ravdess_authorizer,
    mayo_authorizer,
):
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    ravdess_inventory, mayo_inventory = inputs_cli._live_privacy_inventories(args)
    return inputs_cli._build_live_forbidden_tokens(
        mayo_roots=(args.mayo_data_root, args.mayo_existing_export_root),
        ravdess_authorization=ravdess_authorizer(),
        mayo_authorization=mayo_authorizer(),
        ravdess_inventory=ravdess_inventory,
        mayo_inventory=mayo_inventory,
    )


def _scan_private_results(roots: tuple[Path, ...], forbidden) -> None:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    modes_ok, privacy_ok, non_0600 = inputs_cli._scan_private_trees(
        roots, forbidden=forbidden,
    )
    if not modes_ok or not privacy_ok or non_0600 != 0:
        raise ValueError("private SSL result verification failed")


def _assert_summary_private(payload: bytes, forbidden) -> None:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    matcher = inputs_cli._ByteMatcher(forbidden.tokens)
    _, leaked = matcher.feed(payload)
    if leaked:
        raise ValueError("private fact entered the SSL summary")


def _quiet_call(function, /, *args, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        value = function(*args, **kwargs)
    if stdout.getvalue() or stderr.getvalue():
        raise ValueError("pretraining dependency emitted unexpected output")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("two-stage")
    command.add_argument("--mode", choices=("smoke", "formal"), required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--bridge-root", type=Path, required=True)
    command.add_argument("--ravdess-data-root", type=Path, required=True)
    command.add_argument("--ravdess-key", type=Path, required=True)
    command.add_argument("--mayo-data-root", type=Path, required=True)
    command.add_argument("--mayo-existing-export-root", type=Path, required=True)
    command.add_argument("--mayo-cache-root", type=Path, required=True)
    command.add_argument("--mayo-exposure-manifest", type=Path, required=True)
    command.add_argument("--mayo-key", type=Path, required=True)
    return parser


def _private_directory(path: Path, name: str) -> Path:
    lexical = path.absolute()
    try:
        resolved = lexical.resolve(strict=True)
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    canonical = resolved == lexical
    if not canonical and sys.platform == "darwin":
        for source, destination in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
        ):
            try:
                relative = lexical.relative_to(source)
            except ValueError:
                continue
            if resolved == destination / relative:
                canonical = True
                break
    if (
        not canonical
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ValueError(f"{name} must be an owner-only canonical directory")
    return resolved


def _require_existing_input(path: Path, *, directory: bool, name: str) -> None:
    """Reject an absent or symlinked live input before any run-side mutation."""
    lexical = path.absolute()
    try:
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    expected = stat.S_ISDIR(status.st_mode) if directory else stat.S_ISREG(
        status.st_mode
    )
    if not expected or lexical.is_symlink():
        raise ValueError(f"{name} has the wrong storage type")


def _preflight_live_inputs(args: argparse.Namespace) -> None:
    for path, name in (
        (args.ravdess_data_root, "RAVDESS data root"),
        (args.mayo_data_root, "Mayo data root"),
        (args.mayo_existing_export_root, "Mayo existing-export root"),
        (args.mayo_cache_root, "Mayo cache root"),
    ):
        _require_existing_input(path, directory=True, name=name)
    for path, name in (
        (args.ravdess_key, "RAVDESS key"),
        (args.mayo_exposure_manifest, "Mayo exposure manifest"),
        (args.mayo_key, "Mayo key"),
    ):
        _require_existing_input(path, directory=False, name=name)


def _anchor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid),
    )


def _validate_run_root(
    descriptor: int,
    run_root: Path,
    expected_identity: tuple[int, ...],
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(run_root, follow_symlinks=False)
    if (
        _anchor_identity(opened) != expected_identity
        or _anchor_identity(current) != expected_identity
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        raise ValueError("run root changed during execution")


def _validate_results_lock(
    descriptor: int,
    lock_name: str,
    run_descriptor: int,
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(
        lock_name, dir_fd=run_descriptor, follow_symlinks=False,
    )
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_gid, value.st_nlink, value.st_size,
    )
    if (
        identity(opened) != identity(current)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or opened.st_size != 0
    ):
        raise ValueError("results lock storage is unsafe")


def _close_descriptor_sequence(descriptors: tuple[int, ...]) -> None:
    """Attempt every close while preserving active and cleanup exceptions."""
    closer = ExitStack()
    for descriptor in descriptors:
        closer.callback(os.close, descriptor)
    closer.__exit__(*sys.exc_info())


@contextmanager
def _exclusive_results_lock(
    run_root: Path,
) -> Iterator[tuple[int, str, int, tuple[int, ...]]]:
    run_descriptor = os.open(
        run_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        run_identity = _anchor_identity(os.fstat(run_descriptor))
        _validate_run_root(run_descriptor, run_root, run_identity)
        lock_name = ".results.lock"
        created = False
        try:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=run_descriptor,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=run_descriptor,
            )
        if created:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.fsync(run_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(descriptor, lock_name, run_descriptor)
        yield descriptor, lock_name, run_descriptor, run_identity
    finally:
        descriptors = (
            (run_descriptor,)
            + ((descriptor,) if descriptor is not None else ())
        )
        _close_descriptor_sequence(descriptors)


def _rename_directory_no_replace(
    source_name: str,
    destination_name: str,
    *,
    parent_descriptor: int,
) -> None:
    """Atomically publish one directory without replacing any destination."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004 | 0x00000010,
        )
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _classify_results_publication(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    expected_identity: tuple[int, int],
) -> str:
    """Classify an atomic rename outcome against the held staging inode."""
    observed: dict[str, tuple[int, int] | None] = {}
    for field, name in (
        ("source", source_name), ("destination", destination_name),
    ):
        try:
            status = os.stat(
                name, dir_fd=parent_descriptor, follow_symlinks=False,
            )
        except FileNotFoundError:
            observed[field] = None
        else:
            observed[field] = (int(status.st_dev), int(status.st_ino))
    if (
        observed["source"] == expected_identity
        and observed["destination"] is None
    ):
        return "staged"
    if (
        observed["destination"] == expected_identity
        and observed["source"] is None
    ):
        return "published"
    return "indeterminate"


def _write_private_json(path: Path, value: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written < 1:
                raise OSError("private JSON write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ledger_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _directory_ledger_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
    )


def _hash_held_file(descriptor: int, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        total += len(chunk)
        if total > expected_size:
            raise ValueError("result file grew during held verification")
        digest.update(chunk)
    if total != expected_size:
        raise ValueError("result file changed size during held verification")
    return digest.hexdigest()


@contextmanager
def _hold_exact_result_tree(
    run_descriptor: int,
    staging_name: str,
    expected_files: set[str],
) -> Iterator[Callable[..., None]]:
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(
        staging_name, directory_flags, dir_fd=run_descriptor,
    )
    directory_descriptors: dict[tuple[str, ...], int] = {
        (): root_descriptor,
    }
    file_descriptors: dict[tuple[str, ...], int] = {}
    identities: dict[tuple[str, ...], tuple[int, ...]] = {}
    digests: dict[tuple[str, ...], str] = {}
    expected_children: dict[tuple[str, ...], set[str]] = {(): set()}
    try:
        file_parts = {tuple(Path(value).parts) for value in expected_files}
        directory_parts: set[tuple[str, ...]] = set()
        for parts in file_parts:
            for size in range(1, len(parts)):
                directory_parts.add(parts[:size])
        if (
            not file_parts
            or 1 + len(directory_parts) + len(file_parts)
            > _MAX_EXACT_RESULT_TREE_ENTRIES
            or max(len(parts) for parts in file_parts)
            > _MAX_EXACT_RESULT_TREE_DEPTH
        ):
            raise ValueError("result tree exceeds its exact structural budget")
        for parts in sorted(directory_parts, key=lambda value: (len(value), value)):
            parent = parts[:-1]
            expected_children.setdefault(parent, set()).add(parts[-1])
            expected_children.setdefault(parts, set())
            descriptor = os.open(
                parts[-1], directory_flags,
                dir_fd=directory_descriptors[parent],
            )
            directory_descriptors[parts] = descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o700
            ):
                raise ValueError("result directory is not owner-only")
        for parts in sorted(file_parts):
            parent = parts[:-1]
            expected_children.setdefault(parent, set()).add(parts[-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptors[parent],
            )
            file_descriptors[parts] = descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
                or status.st_nlink != 1
            ):
                raise ValueError("result file is not private regular storage")
            identities[parts] = _ledger_identity(status)
        total_regular_bytes = sum(
            int(identity[6]) for identity in identities.values()
        )
        if total_regular_bytes > _MAX_EXACT_RESULT_TREE_REGULAR_BYTES:
            raise ValueError("result tree exceeds its regular-payload budget")
        for parts, descriptor in file_descriptors.items():
            digests[parts] = _hash_held_file(
                descriptor, int(identities[parts][6]),
            )
        for parts, descriptor in directory_descriptors.items():
            identities[parts] = _directory_ledger_identity(
                os.fstat(descriptor)
            )

        def validate(live_root_name: str = staging_name) -> None:
            for parts, descriptor in directory_descriptors.items():
                parent_descriptor = (
                    run_descriptor
                    if not parts
                    else directory_descriptors[parts[:-1]]
                )
                name = live_root_name if not parts else parts[-1]
                live = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False,
                )
                if (
                    _directory_ledger_identity(os.fstat(descriptor))
                    != identities[parts]
                    or _directory_ledger_identity(live) != identities[parts]
                    or set(os.listdir(descriptor))
                    != expected_children.get(parts, set())
                ):
                    raise ValueError("result directory changed before publication")
            for parts, descriptor in file_descriptors.items():
                live = os.stat(
                    parts[-1],
                    dir_fd=directory_descriptors[parts[:-1]],
                    follow_symlinks=False,
                )
                expected = identities[parts]
                if (
                    _ledger_identity(os.fstat(descriptor)) != expected
                    or _ledger_identity(live) != expected
                    or _hash_held_file(descriptor, expected[6]) != digests[parts]
                ):
                    raise ValueError("result file changed before publication")

        validate()
        yield validate
    finally:
        directory_order = tuple(
            directory_descriptors[parts]
            for parts in sorted(
                directory_descriptors, key=lambda value: (len(value), value),
            )
        )
        _close_descriptor_sequence(
            directory_order + tuple(file_descriptors.values())
        )


def _stage_report(result, checkpoint_fingerprint: str) -> dict[str, object]:
    report = result.heldout_report
    value: dict[str, object] = {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "optimizer_steps": result.training_receipt.optimizer_steps,
        "seed": result.training_receipt.seed,
    }
    if result.training_receipt.prior_checkpoint_sha256 is not None:
        value["prior_checkpoint_fingerprint"] = (
            result.training_receipt.prior_checkpoint_sha256
        )
    if result.stage_evidence.mode == "smoke":
        value["train_loss"] = report["train_loss"]
    else:
        value["reconstruction"] = dict(report)
    return value


def _formal_aggregates(
    execution: list[dict[str, object]],
) -> dict[str, object]:
    metrics = {
        "ravdess_only": ("trained", "untrained", "train_mean"),
        "ravdess_then_mayo": (
            "trained", "prior_ravdess", "fresh_untrained", "train_mean",
        ),
    }
    aggregate: dict[str, object] = {}
    for stage, names in metrics.items():
        stage_aggregate: dict[str, object] = {}
        for name in names:
            values = [
                float(run[stage]["reconstruction"][name])  # type: ignore[index]
                for run in execution
            ]
            if len(values) != 3 or not all(
                value >= 0.0 and value < float("inf") for value in values
            ):
                raise ValueError("formal reconstruction evidence is incomplete")
            stage_aggregate[name] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values),
            }
        aggregate[stage] = stage_aggregate
    return aggregate


def _expected_result_files(mode: str, seeds: tuple[int, ...]) -> set[str]:
    if mode not in {"smoke", "formal"}:
        raise ValueError("result file mode is unsupported")
    expected: set[str] = set()
    for seed in seeds:
        prefix = "checkpoints" if mode == "smoke" else (
            f"checkpoints/seed_{seed}"
        )
        for name in ("ravdess_only.pt", "ravdess_then_mayo.pt"):
            expected.add(f"{prefix}/{name}")
            expected.add(f"{prefix}/{name}.receipt.json")
    report_name = (
        "execution_only.json"
        if mode == "smoke"
        else "formal_pretraining_results.json"
    )
    expected.add(f"reports/{report_name}")
    return expected


def _publish_validated_results(
    *,
    run_descriptor: int,
    run_root: Path,
    run_identity: tuple[int, ...],
    results_lock: int,
    lock_name: str,
    staging_name: str,
    staged_identity: tuple[int, int],
    inputs_root: Path,
    privacy_forbidden,
    validate_result_tree: Callable[..., None],
) -> None:
    """Publish and classify one exact result tree through held descriptors."""
    staging = run_root / staging_name
    validate_result_tree()
    _quiet_call(
        _scan_private_results,
        (inputs_root, staging),
        privacy_forbidden,
    )
    validate_result_tree()
    _validate_run_root(run_descriptor, run_root, run_identity)
    _validate_results_lock(results_lock, lock_name, run_descriptor)
    os.fsync(run_descriptor)
    validate_result_tree()
    rename_error: BaseException | None = None
    try:
        _rename_directory_no_replace(
            staging_name,
            "results",
            parent_descriptor=run_descriptor,
        )
    except BaseException as caught:
        rename_error = caught
    publication_state = _classify_results_publication(
        run_descriptor,
        staging_name,
        "results",
        staged_identity,
    )
    if publication_state != "published":
        if publication_state == "staged" and rename_error is not None:
            raise rename_error
        raise RuntimeError(
            "results publication outcome is indeterminate"
        ) from rename_error
    try:
        # A faulting wrapper may report failure after the syscall committed.
        # Success requires the canonical name, exact held tree, privacy closure,
        # lock identity, and parent durability all to revalidate.
        validate_result_tree("results")
        _quiet_call(
            _scan_private_results,
            (inputs_root, run_root / "results"),
            privacy_forbidden,
        )
        validate_result_tree("results")
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(results_lock, lock_name, run_descriptor)
        os.fsync(run_descriptor)
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(results_lock, lock_name, run_descriptor)
        validate_result_tree("results")
        _quiet_call(
            _scan_private_results,
            (inputs_root, run_root / "results"),
            privacy_forbidden,
        )
        validate_result_tree("results")
    except BaseException as caught:
        raise RuntimeError(
            "published results are retained as indeterminate evidence"
        ) from caught
    if rename_error is not None:
        raise RuntimeError(
            "published results are retained after a rename return fault"
        ) from rename_error


def _run_two_stage(args: argparse.Namespace) -> dict[str, object]:
    run_root = _private_directory(args.run_root.absolute(), "run root")
    if args.mode == "smoke":
        namespace_root = run_root.parent.parent
        namespace_ok = (
            run_root.parent.name == "smoke"
            and _RUN_ID.fullmatch(run_root.name) is not None
        )
    else:
        namespace_root = run_root.parent
        namespace_ok = run_root.name == "formal"
    if (
        not namespace_ok
        or namespace_root != PRETRAINING_ROOT.absolute()
    ):
        raise ValueError("run root is outside the exact mode/run-id namespace")
    inputs_root = _private_directory(run_root / "inputs", "frozen inputs")
    bridge_root = _private_directory(
        args.bridge_root.absolute(), "bridge generation"
    )
    if bridge_root != (PRETRAINING_ROOT / "bridge").absolute():
        raise ValueError("bridge root is outside the canonical namespace")
    _preflight_live_inputs(args)
    ravdess_authorizer, mayo_authorizer = _quiet_call(
        _authorization_factories, args,
    )
    producer_sha256 = _quiet_call(_producer_sha256)
    privacy_forbidden = _quiet_call(
        _privacy_forbidden,
        args, ravdess_authorizer, mayo_authorizer,
    )
    seeds = (0,) if args.mode == "smoke" else (0, 1, 2)
    summary = {
        "checkpoint_count": len(seeds) * 2,
        "mode": args.mode,
        "seed_count": len(seeds),
        "stage_count": 2,
    }
    _assert_summary_private(
        json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii"),
        privacy_forbidden,
    )
    with _exclusive_results_lock(run_root) as (
        results_lock, lock_name, run_descriptor, run_identity,
    ):
        names = set(os.listdir(run_descriptor))
        if "results" in names:
            raise FileExistsError("results generation already exists")
        if any(name.startswith(".results.staging-") for name in names):
            raise ValueError("unresolved results staging state blocks execution")

        ravdess_evidence = _quiet_call(
            ssl_core.authorize_frozen_ssl_stage,
            stage="ravdess",
            mode=args.mode,
            inputs_root=inputs_root,
            bridge_root=bridge_root,
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer_sha256,
        )
        staging = run_root / f".results.staging-{secrets.token_hex(16)}"
        staging.mkdir(mode=0o700)
        os.chmod(staging, 0o700)
        staged_status = staging.stat()
        staged_identity = (
            staged_status.st_dev,
            staged_status.st_ino,
        )
        checkpoints = staging / "checkpoints"
        reports = staging / "reports"
        checkpoints.mkdir(mode=0o700)
        reports.mkdir(mode=0o700)
        os.chmod(checkpoints, 0o700)
        os.chmod(reports, 0o700)

        execution: list[dict[str, object]] = []
        reload_checks: list[tuple[Path, object, object, str]] = []
        retained_runtime: list[object] = []
        checkpoint_directories = {checkpoints}
        for seed in seeds:
            seed_checkpoints = checkpoints
            if args.mode == "formal":
                seed_checkpoints = checkpoints / f"seed_{seed}"
                seed_checkpoints.mkdir(mode=0o700)
                os.chmod(seed_checkpoints, 0o700)
                checkpoint_directories.add(seed_checkpoints)
            ravdess_result = _quiet_call(
                ssl_core.train_ssl_stage,
                stage_evidence=ravdess_evidence,
                seed=seed,
            )
            ravdess_payload = _quiet_call(
                ssl_core.build_ssl_checkpoint_payload, ravdess_result,
            )
            ravdess_path = seed_checkpoints / "ravdess_only.pt"
            ravdess_receipt = _quiet_call(
                ssl_core.save_ssl_checkpoint,
                ravdess_path,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            persisted_ravdess = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                ravdess_path,
                receipt=ravdess_receipt,
                stage_evidence=ravdess_evidence,
            )
            retained_runtime.extend((ravdess_receipt, persisted_ravdess))
            ravdess_fingerprint = _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, persisted_ravdess,
            )
            reload_checks.append((
                ravdess_path, ravdess_receipt, ravdess_evidence,
                ravdess_fingerprint,
            ))

            mayo_evidence = _quiet_call(
                ssl_core.authorize_frozen_ssl_stage,
                stage="mayo",
                mode=args.mode,
                inputs_root=inputs_root,
                bridge_root=bridge_root,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer_sha256,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_ravdess_evidence=ravdess_evidence,
            )
            retained_runtime.append(mayo_evidence)
            mayo_result = _quiet_call(
                ssl_core.train_ssl_stage,
                stage_evidence=mayo_evidence,
                seed=seed,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_stage_evidence=ravdess_evidence,
            )
            mayo_payload = _quiet_call(
                ssl_core.build_ssl_checkpoint_payload, mayo_result,
            )
            mayo_path = seed_checkpoints / "ravdess_then_mayo.pt"
            mayo_receipt = _quiet_call(
                ssl_core.save_ssl_checkpoint,
                mayo_path,
                mayo_payload,
                stage_evidence=mayo_evidence,
            )
            persisted_mayo = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                mayo_path,
                receipt=mayo_receipt,
                stage_evidence=mayo_evidence,
            )
            retained_runtime.extend((mayo_receipt, persisted_mayo))
            mayo_fingerprint = _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, persisted_mayo,
            )
            reload_checks.append((
                mayo_path, mayo_receipt, mayo_evidence, mayo_fingerprint,
            ))
            execution.append({
                "seed": seed,
                "ravdess_only": _stage_report(
                    ravdess_result, ravdess_fingerprint,
                ),
                "ravdess_then_mayo": _stage_report(
                    mayo_result, mayo_fingerprint,
                ),
            })

        report = {
            "schema_version": "dynamic_landmark_ssl_execution_report_v1",
            "mode": args.mode,
            "seed_count": len(seeds),
            "stage_count": 2,
            "runs": execution,
            "medical_generalization": False,
            "outer_mayo_predictions_viewed": False,
        }
        if args.mode == "formal":
            report["aggregate"] = _formal_aggregates(execution)
        report_name = (
            "execution_only.json"
            if args.mode == "smoke"
            else "formal_pretraining_results.json"
        )
        report_path = reports / report_name
        _quiet_call(_write_private_json, report_path, report)
        (
            _report_resolved,
            report_bytes,
            _report_sha256,
            _report_identity,
        ) = _quiet_call(
            ssl_core._private_regular_file_snapshot,
            report_path,
            "SSL execution report",
        )
        report_value = _quiet_call(
            ssl_core._strict_json_mapping,
            report_bytes,
            "SSL execution report",
        )
        if not ssl_core._exact_json_value(report_value, report):
            raise ValueError("SSL execution report changed after writing")
        for path, receipt, evidence, expected in reload_checks:
            reloaded = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                path, receipt=receipt, stage_evidence=evidence,
            )
            if _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, reloaded,
            ) != expected:
                raise ValueError("checkpoint changed during result finalization")
        expected_files = _expected_result_files(args.mode, seeds)
        with _hold_exact_result_tree(
            run_descriptor, staging.name, expected_files,
        ) as validate_result_tree:
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging), privacy_forbidden,
            )
            for checkpoint_directory in sorted(
                checkpoint_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _quiet_call(_fsync_directory, checkpoint_directory)
            _quiet_call(_fsync_directory, reports)
            _quiet_call(_fsync_directory, staging)
            if "results" in set(os.listdir(run_descriptor)):
                raise FileExistsError(
                    "results generation appeared during execution"
                )
            before_publish = os.stat(
                staging.name,
                dir_fd=run_descriptor,
                follow_symlinks=False,
            )
            if (before_publish.st_dev, before_publish.st_ino) != staged_identity:
                raise ValueError(
                    "results staging identity changed before publication"
                )
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging), privacy_forbidden,
            )
            validate_result_tree()
            _validate_run_root(run_descriptor, run_root, run_identity)
            _validate_results_lock(
                results_lock, lock_name, run_descriptor,
            )
            os.fsync(run_descriptor)
            _publish_validated_results(
                run_descriptor=run_descriptor,
                run_root=run_root,
                run_identity=run_identity,
                results_lock=results_lock,
                lock_name=lock_name,
                staging_name=staging.name,
                staged_identity=staged_identity,
                inputs_root=inputs_root,
                privacy_forbidden=privacy_forbidden,
                validate_result_tree=validate_result_tree,
            )

    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.command != "two-stage":
        raise ValueError("unsupported pretraining command")
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    captured = inputs_cli._run_mayo_cli_captured(
        args, lambda: _run_two_stage(args),
    )
    print(captured.json_line)
    result = json.loads(captured.json_line)
    if type(result) is not dict:
        raise ValueError("pretraining result is malformed")
    return result


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, OSError, RuntimeError, ValueError):
        print("dynamic landmark pretraining failed closed", file=sys.stderr)
        raise SystemExit(2)
